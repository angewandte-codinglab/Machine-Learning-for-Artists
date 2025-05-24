from deepface import DeepFace
import cv2

reference_image_path = "images/IMG_1091.jpeg"

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
       
        x, y, w, h = faces[0]
        detected_face = frame[y:y+h, x:x+w]

        detected_face_path = "temp_face.jpg"
        cv2.imwrite(detected_face_path, detected_face)

        try:
            result = DeepFace.verify(reference_image_path, detected_face_path, model_name="Facenet")
            if result["verified"]:
                label = "Recognized"
                color = (0, 255, 0)  # Green for recognized
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown
        except Exception as e:
            label = "Error"
            color = (0, 0, 255)
            print(f"Error during verification: {e}")
    else:
        label = "No Face Detected"
        color = (0, 0, 255)


    if len(faces) > 0:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    cv2.imshow("Face Recognition", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()