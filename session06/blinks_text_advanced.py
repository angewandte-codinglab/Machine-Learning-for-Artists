import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

FACIAL_LANDMARKS_2D = [33, 263, 1, 61, 291, 199] 

FACIAL_LANDMARKS_3D = np.array([
    [0.0, 0.0, 0.0],       
    [-30.0, -65.5, -5.0],   
    [30.0, -65.5, -5.0],    
    [-60.0, -40.0, -20.0], 
    [60.0, -40.0, -20.0],  
    [0.0, -120.0, -10.0]
], dtype=np.float64)

def normalize_angle(angle):
    angle = angle % 360
    if angle > 180:
        angle -= 360
    return angle


def eye_opening_ratio(eye_points):
    vertical1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical2 = np.linalg.norm(eye_points[2] - eye_points[4])
    vertical = (vertical1 + vertical2) / 2.0
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])
    return vertical / horizontal

def get_eye_points(landmarks, idxs, w, h):
    return np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in idxs])

def get_head_pose(landmarks, w, h):
    image_points = np.array([
        (landmarks[33].x * w, landmarks[33].y * h),   
        (landmarks[1].x * w, landmarks[1].y * h),     
        (landmarks[263].x * w, landmarks[263].y * h), 
        (landmarks[61].x * w, landmarks[61].y * h),   
        (landmarks[291].x * w, landmarks[291].y * h), 
        (landmarks[199].x * w, landmarks[199].y * h),
    ], dtype=np.float64)
    
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4,1))  

    success, rotation_vector, translation_vector = cv2.solvePnP(
        FACIAL_LANDMARKS_3D, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vector)
    proj_matrix = np.hstack((rmat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    pitch, yaw, roll = euler_angles.flatten()
    return pitch, yaw, roll

cap = cv2.VideoCapture(0)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

EAR_THRESHOLD = 0.25
MIN_PITCH_DEG = 130

while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    blink_text = ""

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye = get_eye_points(landmarks, LEFT_EYE_IDX, w, h)
        right_eye = get_eye_points(landmarks, RIGHT_EYE_IDX, w, h)

        left_ratio = eye_opening_ratio(left_eye)
        right_ratio = eye_opening_ratio(right_eye)
        avg_ratio = (left_ratio + right_ratio) / 2

        head_pose = get_head_pose(landmarks, w, h)
        if head_pose is not None:
            pitch, yaw, roll = head_pose

            if abs(pitch) > MIN_PITCH_DEG and avg_ratio < EAR_THRESHOLD:
                blink_text = "Blink!"

            cv2.putText(frame, f"Pitch: {pitch:.1f}", (10, h - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Yaw: {yaw:.1f}", (10, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Roll: {roll:.1f}", (10, h - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            if avg_ratio < EAR_THRESHOLD:
                blink_text = "Blink!"

        for pt in np.concatenate([left_eye, right_eye]):
            cv2.circle(frame, tuple(pt.astype(int)), 2, (0, 255, 0), -1)

        cv2.putText(frame, f"Eye ratio: {avg_ratio:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        if blink_text:
            cv2.putText(frame, blink_text, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Blink Detection with Head Pose Filter", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
