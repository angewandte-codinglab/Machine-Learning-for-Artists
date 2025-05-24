# Session 06

## 1. DeepFace

We have noticed, that a reliable detection of the face (especially in different environments/surroundings/lighting situations) is only possible with a vast amount of data. Even One-Shot systems like YOLOE fail more or less detecting the same face reliably. 

For face detection we can use [Deepface](https://github.com/serengil/deepface) to detect faces only using a single reference image.

<br><br><br>

### 1.1 Setup

1. Create a new folder on your computer, name it accordingly

2. Open the folder in VS Code and create a virtual environment, as we did with YOLO. **Important**: Make sure you use a Python Version from 3.7-3.10. Newer versions will not work!

3. In session06 navigate to folder `Deepface` and download the [deepface_requirements.txt](deepface_requirements.txt).

4. Place the deepface_requirements.txt file in your folder 

5. Install requirements with: 
    ```bash
    pip install -r deepface_requirements.txt
    pip install deepface
    ```

<br><br><br>
    
### 1.2 Basic Example

For this basic example we try to read out our webcam to detect on faces. The `DeepFace.analyze` can detect (problematically) emotion, age, gender and race. 

1. Create a new python script, call it `basic.py`

2. Import your libraries
    ```python
    from deepface import DeepFace
    import cv2
    ```

3. Initialize your Webcam – a true classic.
    ```python
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    ```

4. Start the main Loop and read frames continuously
    ```python
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
    ```

5. Try to get results. In the `actions` array we can define what should be detected. Print the results to see which values are returned.
    ```python
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'], # age, gender, race
            enforce_detection=False, 
            detector_backend='opencv' # Alternatives 'mtcnn' or 'retinaface' are more accurate but slower
        )
    ```

6. Using a for loop, we read the data for every time we get a new result. 
    ```python
    for face_data in results:
    ```

7. Read out dominant_emotion and region values. This returns the strongest emotion and the position of the face.
    ```python
    dominant_emotion = face_data.get('dominant_emotion')
    region = face_data.get('region')
    ```

8. Draw the detections.
    ```python
    if dominant_emotion and region:
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        text = f"{dominant_emotion}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    else:
        print(f"Error")


9. Show frame and exit functions
    ```python
        except Exception as e:
            pass 

        cv2.imshow('Deepface', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```




<details>
<summary>Full Code</summary>

```python
import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=['emotion'], # age, gender
            enforce_detection=False, 
            detector_backend='opencv' # Faster face detector; 'mtcnn' or 'retinaface' are more accurate but slower
        )

        if isinstance(results, list) and len(results) > 0:
            for face_data in results:
                if isinstance(face_data, dict): 
                    dominant_emotion = face_data.get('dominant_emotion')
                    region = face_data.get('region') # {'x': X, 'y': Y, 'w': W, 'h': H}

                    if dominant_emotion and region:
                        x, y, w, h = region['x'], region['y'], region['w'], region['h']
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                        
                        text = f"{dominant_emotion}"
                        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    print(f"Unexpected result format from DeepFace: {face_data}")

    except Exception as e:
        pass 

    cv2.imshow('Emotion Detection (DeepFace)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam released and windows closed.")

```

</details>

<br><br><br>

### 1.3 Verify Faces

By using one reference image of a face, it is possible to detect the same face in a webcam stream 

1. Like in the example 1.2 above, we define the imports and our webcam. We only add one line before the while loop to specify our `reference_image`. This is an image of your face and should be loaded into VS Code. 

    ```python
     reference_image_path = "images/foto.jpeg"
    ```

    Our first chunk of code looks like this:
     ```python
    from deepface import DeepFace
    import cv2

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        exit()

    reference_image = cv2.imread("images/foto.jpeg")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

    region = None
    label = "Unknown"
    color = (0, 0, 255)  # Default color for unknown

    ```

2. Try to read out the results:
    ```python
    try:
        result = DeepFace.verify(reference_image, frame, model_name="Facenet", detector_backend="opencv")
        region = result.get("facial_areas", {}).get("img2")
    ```

3. Set the values for the results to be used in the draw function bleow
    ```python
    if result["verified"]:
        label = "Recognized"
        color = (0, 255, 0)
    else:
        label = "Unknown"
        color = (0, 0, 255)
    ```

4. Add error exception

    ```python
    except Exception as e:
        print(f"Error during verification: {e}")
    ````

5. Draw the results on the frame

    ```python
    if region:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    ````

6. Show frame and exit functions

    ```python
        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    ```
        
<br><br><br>

<details>
<summary>Full Code</summary>

```python
from deepface import DeepFace
import cv2

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

reference_image = cv2.imread("images/foto.jpeg")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    try:
        result = DeepFace.verify(reference_image, frame, model_name="Facenet", detector_backend="opencv")
        region = result.get("facial_areas", {}).get("img2")
        if result["verified"]:
            label = "Recognized"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)
    except Exception as e:
        print(f"Error during verification: {e}")
        

    
    if region:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

### 1.4 Send an OSC Message when a face is verified

1. Implement the logic from the OSC Send script and send to my ip a string when a face is verified

2. Extra: Also send the position of the face on the frame via OSC.

<br><br><br>

### 1.5 Detect your favourite politician

1. Find an image depicting your (least) favourite politician

2. Detect this person in a YouTube video?


<br><br><br>

### 1.6 The Real Flamish Scrollers!
With the techniques we gathered, we can now write a script that detects a specific person and a phone. For this we need to incorporate the YOLO Logic into our deepface script.

1. Add Ultralytics to your imports. You will need to install it again via `pip install ultralytics`
    ```python
    from deepface import DeepFace
    import cv2
    from ultralytics import YOLO
    ```

2. Define your YOLO model in your main declarations (right after face_cascade)
    ```python
    yolo_model = YOLO("yolo11n.pt")
    ````

3. After the face verification loop (if len(faces)...) add the YOLO Logic.
    ```python 
    results = yolo_model(frame, classes=[67], verbose=False)
    ```
    Then, draw the Yolo Boxes
    ```python
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = yolo_model.model.names[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    ```

<br>

Next, lets add Logic that saves a photo when both of the classes are present in the frame.

4. Just before your DeepFace loop (if len(faces)...), add define a bool. When this bool changes to `true` (and another one for YOLO) the photo will be taken.

5. In the verified if statement add
    ```python
    face_verified = True
    ```

6. The same goes for the YOLO Detection Loop. Before the line `results = ...` add
    ```python
    phone_detected = False
    ```

7. In the loop add 
    ```python
    phone_detected = True
    ```

8. Save a photo – Below your YOLO Detection write a condition that checks if `face_verified` and `phone_detected` are both true

    ```python
    if face_verified and phone_detected:
        cv2.imwrite("detected_face_and_phone.jpg", frame)
    ```


<br><br>

<details>
<summary>Full Code</summary>

```python
from deepface import DeepFace
import cv2
from ultralytics import YOLO


reference_image_path = "images/foto.jpeg"

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

yolo_model = YOLO("yolo11n.pt")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # --- DeepFace Face Recognition ---
    face_verified = False
    label = "No Face Detected"
    color = (0, 0, 255)
    region = None

    try:
        result = DeepFace.verify(reference_image_path, frame, model_name="Facenet", detector_backend="opencv")
        region = result.get("facial_areas", {}).get("img2")
        if result["verified"]:
            label = "Recognized"
            color = (0, 255, 0)
            face_verified = True
        else:
            label = "Unknown"
            color = (0, 0, 255)
    except Exception as e:
        label = "Error"
        color = (0, 0, 255)
        print(f"Error during verification: {e}")

    if region:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    else:
        cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # --- YOLO Object Detection ---
    phone_detected = False
    results = yolo_model(frame, classes=[67], verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = yolo_model.model.names[cls]
            
            phone_detected = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    if face_verified and phone_detected:
        cv2.imwrite("detected_face_and_phone.jpg", frame)

    cv2.imshow("The Flamish Scrollers", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>



## 2. Mediapipe

Mediapipe is a framework by Google for real-time computer vision applications. It's good at looking at your body for use cases like **hand tracking, face mesh, pose estimation, object detection and holistic body tracking**. 

### 2.1 Setup

1. Create a folder on your computer dedicated to mediapipe.

2. Create a virtual environment

3. Install mediapipe via pip

    ```bash
    pip install mediapipe opencv-python numpy
    ```

<br><br><br>

### 2.2 Simple Hand Tracking

1. Import opencv and mediapipe and define webcam
    ```python
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(0)
    ```

2. Define what we want to use from mediapipe. We use `hands` and `drawing_utils`
    ```python
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    ```

3. Setup the parameters for the hands detector
    ```python
    hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
    )
    ```

4. While functions when reading from webcam
    ```python
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    ```

5. Flip the webcam image (in most cases necessary), read results
    ```python
    frame = cv2.flip(frame, 1)
    results = hands.process(frame)
    ```

6. Read results and draw landmarks on the frame
    ```python
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    ```

7. Show image and break functions
    ```python
        cv2.imshow('Hands', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    ```

<br>
<details>
<summary>Full Code</summary>

```python
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow('Hands', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

### 2.3 Hand Zoom Example

We can calculate the distance between fingers with this code. Find out which landmarks the fingers have [here.](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker#models) The ends of pinky and thumb are 20 and 4. The `distance` parameter drives the zoom of the image here. 

1. Add Numpy to to the imports, we need it for the calculation of the distance.
    ```python
    import cv2
    import mediapipe as mp
    import numpy as np

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    ```

2. Define a function that gets the position of the fingertips relative to the frame widht & height. `np.linalg.norm` is used to calculate the distance between points. The calculated value is the returned.
    ```python
    def get_pinky_thumb_distance(hand_landmarks, frame_width, frame_height):
    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]
    x1, y1 = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
    x2, y2 = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)
    return np.linalg.norm([x2 - x1, y2 - y1])


3. Call the mediapipe hands and add initial values for zoom and base_distance
    ```python
        hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    zoom = 1.0
    base_distance = None
    ```

4. Read and flip frame, get results from detector
    ```python
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        frame = cv2.flip(frame, 1)
        results = hands.process(frame)
    ```

5. Get dimemsion of the video frame
   ```python
   h, w = frame.shape[:2]
   ```

6. Check if there is output from the detector. If yes, then draw the connections on the frame using the `mp_drawing` function.
    ```python
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    ```

7. Call our function from before to calculate the distances between the fingers.
    ```python
    distance = get_pinky_thumb_distance(hand_landmarks, w, h)
    ```

8. Set the distance of the first calculation as base. Then divide distance with base distance. This defines our zoom value. Utilising the np.clip function we can specify min and max values.
    ```python
    if base_distance is None:
            base_distance = distance if distance > 0 else 1
        
    zoom = np.clip(distance / base_distance, 1.0, 3.0)
    ```

9. Draw a line between the fingers
    ```python
    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]
    pt1 = (int(thumb_tip.x * w), int(thumb_tip.y * h))
    pt2 = (int(pinky_tip.x * w), int(pinky_tip.y * h))
    cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
    cv2.putText(frame, f"Zoom: {zoom:.2f}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    ```

10. Set zoom to 1.0 when no hands are detected
    ```python
    else:
    base_distance = None
    zoom = 1.0
    ```

11. With the `zoom` value, set the zoom. We make sure this is only excecuted when the zoom is above 1.0. 
    ```python
    if zoom > 1.0:
        center_x, center_y = w // 2, h // 2 # Calculate the centerpoints of the frame
        new_w, new_h = int(w / zoom), int(h / zoom) # Set new zoom (crop) levels for width and height
        x1 = max(center_x - new_w // 2, 0)  # Calculate corner points for cropping
        y1 = max(center_y - new_h // 2, 0)
        x2 = min(center_x + new_w // 2, w)
        y2 = min(center_y + new_h // 2, h)
        cropped = frame[y1:y2, x1:x2] #Crop
        frame = cv2.resize(cropped, (w, h))
    ```

12. Exit Functions
    ```python
        cv2.imshow('MediaPipe Hands Zoom', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
 
    cap.release()
    cv2.destroyAllWindows()   
    ```

<br>

<details>
<summary>Full Code</summary>

```python
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def get_pinky_thumb_distance(hand_landmarks, frame_width, frame_height):
    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]
    x1, y1 = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
    x2, y2 = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)
    return np.linalg.norm([x2 - x1, y2 - y1])

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
zoom = 1.0
base_distance = None

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    frame = cv2.flip(frame, 1)
    results = hands.process(frame)

    h, w = frame.shape[:2]

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
            frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        distance = get_pinky_thumb_distance(hand_landmarks, w, h)
        if base_distance is None:
            base_distance = distance if distance > 0 else 1
        
        zoom = np.clip(distance / base_distance, 1.0, 3.0)

        
        thumb_tip = hand_landmarks.landmark[4]
        pinky_tip = hand_landmarks.landmark[20]
        pt1 = (int(thumb_tip.x * w), int(thumb_tip.y * h))
        pt2 = (int(pinky_tip.x * w), int(pinky_tip.y * h))
        cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        cv2.putText(frame, f"Zoom: {zoom:.2f}x", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        base_distance = None
        zoom = 1.0

    if zoom > 1.0:
        center_x, center_y = w // 2, h // 2
        new_w, new_h = int(w / zoom), int(h / zoom)
        x1 = max(center_x - new_w // 2, 0)
        y1 = max(center_y - new_h // 2, 0)
        x2 = min(center_x + new_w // 2, w)
        y2 = min(center_y + new_h // 2, h)
        cropped = frame[y1:y2, x1:x2]
        frame = cv2.resize(cropped, (w, h))

    cv2.imshow('MediaPipe Hands Zoom', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break

cap.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

### 2.4 Blink Recognition

Using the Face Mesh, we can find out if a person blinks. This works by finding out the distance of the eye landmarks and thus calculating the eye aspect ratio (EAR) with them. The landmarks for the face mesh can be found [here.](https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png)

1. Import libraries. One is new to us: `scipy` is used here to calculate euclidean distances. 
    ```python
    import cv2
    import numpy as np
    import mediapipe as mp
    from scipy.spatial import distance
    ```

2. From the landmarks we define the points for the eyes in an array.
    ```python
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    ```

3. Eye Aspect Ratio Calculation (EAR). A and B: vertical distances between upper and lower eyelids. C: horizontal distance between corners of the eye. `ear` is high when eyes are open and drops when eyes close.
    ```python
    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    ```

4. Set global variables. `EAR_THRESHOLD`: If EAR falls below this, the eye is likely closed. `CONSEC_FRAMES`: Blink is counted only if eye is closed for at least this many consecutive frames. 
    ```python
    EAR_THRESHOLD = 0.25
    CONSEC_FRAMES = 2
    blink_count = 0
    frame_counter = 0
    ```

5. Mediapipe Face Mesh, Drawing and Webcam
    ```python
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    ```

6. Read frames from webcam, get frame dimensions
    ```python
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        result = face_mesh.process(frame)
        h, w = frame.shape[:2]
    ```

7. Get the facial landmarks of the first detected face.
    ```python
    if result.multi_face_landmarks:
        mesh_points = result.multi_face_landmarks[0].landmark
    ```

8. Convert normalized values into pixel values.
    ```python
    left_eye = [np.array([mesh_points[i].x * w, mesh_points[i].y * h]) for i in LEFT_EYE]
    right_eye = [np.array([mesh_points[i].x * w, mesh_points[i].y * h]) for i in RIGHT_EYE]
    ```

9. Call our distance calculation function and get the average of both eyes.
    ```python
    left_ear = eye_aspect_ratio(left_eye)
    right_ear = eye_aspect_ratio(right_eye)
    ear = (left_ear + right_ear) / 2.0
    ```

10. Draw the points for each landmark
    ```python
    for pt in left_eye + right_eye:
        cv2.circle(frame, tuple(np.int32(pt)), 2, (0, 255, 0), -1)
    ```

11. If the EAR value is below the set `EAR_THRESHOLD` add to the frame counter. Counts a blink if the eye was closed for enough frames, then resets.
    ```python
    if ear < EAR_THRESHOLD:
        frame_counter += 1
    else:
        if frame_counter >= CONSEC_FRAMES:
            blink_count += 1
        frame_counter = 0

12. Draw info text
    ```python
    cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    ```

13. Exit functions
    ```python
        cv2.imshow("Blink Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```

<br>

<details>
<summary>Full Code</summary>

```python

import cv2
import numpy as np
import mediapipe as mp
from scipy.spatial import distance



LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 2
blink_count = 0
frame_counter = 0

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    result = face_mesh.process(frame)
    h, w = frame.shape[:2]

    if result.multi_face_landmarks:
        mesh_points = result.multi_face_landmarks[0].landmark
        left_eye = [np.array([mesh_points[i].x * w, mesh_points[i].y * h]) for i in LEFT_EYE]
        right_eye = [np.array([mesh_points[i].x * w, mesh_points[i].y * h]) for i in RIGHT_EYE]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        for pt in left_eye + right_eye:
            cv2.circle(frame, tuple(np.int32(pt)), 2, (0, 255, 0), -1)

        if ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= CONSEC_FRAMES:
                blink_count += 1
            frame_counter = 0

        cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Blink Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

</details>

<br>

#### Advanced script
This blink detection works, but can be cheated, by tilting the head back. When doing this the eyes will look more squinted and possibly fall below the threshold. To prevent this, we can read out the 3D position of the head. Check out the advanced script [blinks_text_advanced.py](blinks_text_advanced.py).

<br><br><br>

### 2.5 Lie Detector
The blink the detection could be used for a lie detector. While researching i found this project on github which can not only detect excessive blinking, but also other markers of lying like: changing of pose, hands before face, heartrate (crazy!)

**[Truthsayer Github by everythingishacked](https://github.com/everythingishacked/Truthsayer)**

1. Download the ZIP / Clone repo into a folder

2. If necessary unzip.

3. Make a new window in VS Code and drag the folder in.

4. Create a virtual environment

5. For me the requirements provided didn't work, so i made some changes. Download [truthsayer_requirements.txt](truthsayer_requirements.txt) and drag it into your VS Code folder.

6. Install requirements via terminal in VS Code
    ```bash
    pip install -r truthsayer_requirements.txt
    ```

7. Run `intercept.py` and see if errors occur. 

<br><br><br>




## 3. Speech Recognition with Whisper

Whisper is a speech to text model that can be used i.e. for live subtitles

### Setup

1. Create a new folder and open it in VS Code

2. Create a virtual environment

3. Install necessary libraries
    ```bash
    pip install openai-whisper sounddevice numpy opencv-python faster-whisper
    ```

<br><br><br>

### 3.1 Basic Example

1. Import dependencies and load the model. The model should download automatically. The weights are quite heavy so it might take a minute depending on which model you choose ("tiny", "base", "small", "medium", "large")
    ```python
    import whisper
    import sounddevice as sd
    import numpy as np

    model = whisper.load_model("small")
    ```

2. Set global values: `duration` for how long it should listen before it resets. `samplerate` of your mic audio. `channels` set to 1 = mono.
    ```python
    duration = 5
    samplerate = 16000
    channels = 1
    ```

3. Define a function for the audio recording
    ```python
    def record_audio(duration, samplerate):
        print("Recording...")
    ```

4. Record the audio for the specified duration. 
   ```python
   audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32')
   ```

5. Wait for the recording to be finished and return flattened audio
    ```python
    sd.wait()
    return audio.flatten()
    ```

6. Try to call our function
    ```python
    try:
        while True:
            audio = record_audio(duration, samplerate)
            print("Transcribing...")
    ```

7. Transcribe the result with model.transcribe, print the result
    ```python
    result = model.transcribe(audio, language='en', fp16=False)
    print("You said:", result["text"].strip())
    ```

8. Exit function
    ```python
    except KeyboardInterrupt:
        print("\n Exiting.")
    ```

<br><br><br>

## 3.2 Live Subtitles

We use the `faster-whisper` library for improved speed. 

```python
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import cv2
import threading
import queue
import time

model_size = "small"
model = WhisperModel(model_size, device="cpu", compute_type="int8")  # Use "cuda" for GPU if available

samplerate = 16000
block_duration = 2  
buffer_duration = 3 
channels = 1

subtitle_text = ""
subtitle_lock = threading.Lock()

audio_buffer = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Audio status:", status)
    audio_chunk = indata.copy().flatten()
    audio_buffer.put(audio_chunk)

def transcribe_stream():
    global subtitle_text
    rolling_audio = np.zeros((0,), dtype=np.float32)

    while True:
        while not audio_buffer.empty():
            rolling_audio = np.concatenate((rolling_audio, audio_buffer.get()))

        max_samples = int(buffer_duration * samplerate)
        if rolling_audio.shape[0] > max_samples:
            rolling_audio = rolling_audio[-max_samples:]

        if rolling_audio.shape[0] >= samplerate: 
            try:
                # faster-whisper expects int16 PCM or float32 numpy array
                segments, _ = model.transcribe(
                    rolling_audio, 
                    language="en", 
                    beam_size=1, 
                    temperature=0.0
                )
                text = ""
                for segment in segments:
                    text += segment.text
                with subtitle_lock:
                    subtitle_text = text.strip()
            except Exception as e:
                print("Transcription error:", e)

        time.sleep(block_duration / 2)

def draw_subtitles(frame, text):
    if not text:
        return frame
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    x = int((frame.shape[1] - text_size[0]) / 2)
    y = frame.shape[0] - 40
    cv2.rectangle(frame, (x - 10, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def main():
    threading.Thread(target=transcribe_stream, daemon=True).start()

    with sd.InputStream(samplerate=samplerate, channels=channels,
                        dtype='float32', callback=audio_callback,
                        blocksize=int(samplerate * block_duration)):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Could not open webcam.")
            return

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            with subtitle_lock:
                display_text = subtitle_text
            frame = draw_subtitles(frame, display_text)

            cv2.imshow("Live Subtitles (Real-Time)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

<br><br><br>


## 3.3 Recognize Trigger Words

```python
import sounddevice as sd
import numpy as np
import queue
import threading
import time
from faster_whisper import WhisperModel

model = WhisperModel("base", compute_type="int8", device="cpu")

samplerate = 16000
block_duration = 1
channels = 1

trigger_words = ["hello", "test", "computer", "python"]

audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print("Status:", status)
    audio_queue.put(indata.copy().flatten())

def recognize_loop():
    rolling_audio = np.zeros((0,), dtype=np.float32)
    buffer_duration = 3 
    max_samples = int(buffer_duration * samplerate)

    while True:
        while not audio_queue.empty():
            rolling_audio = np.concatenate((rolling_audio, audio_queue.get()))

        if rolling_audio.shape[0] > max_samples:
            rolling_audio = rolling_audio[-max_samples:]

        if rolling_audio.shape[0] >= samplerate:  
            segments, _ = model.transcribe(rolling_audio, language="en", beam_size=1)
            full_text = " ".join(segment.text.lower() for segment in segments)

            for word in trigger_words:
                if word in full_text:
                    print(f"Trigger word detected: '{word}'")

        time.sleep(block_duration / 2)

def main():
    threading.Thread(target=recognize_loop, daemon=True).start()

    with sd.InputStream(samplerate=samplerate, channels=channels, dtype='float32',
                        callback=audio_callback, blocksize=int(samplerate * block_duration)):
        print("Listening for trigger words.")
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stopping...")

if __name__ == "__main__":
    main()
```

