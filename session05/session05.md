# Session 05

## 1. Training and Labeling

### 1.1 What does a (Yolo) training set look like?
- Consists of a lot of representative images of the objects/things the algorithm should detect.
- Normally, results start to get good at 800+ images per class (=type of object)
- For each image, there is a corresponing 'label' file, which holds the information on the class (=what object) and its position on the image (=coordinates)
- The structure of a YOLO Dataset typically looks like this:
- - A folder `train` containing 80% of the files:
- - - folder `images` with image files (.jpg, .png)
- - - folder `labels` with corresponding annotation files (.txt)
- - A folder `val` containing 20% of the files:
- - - folder `images` with image files (.jpg, .png)
- - - folder `labels` with corresponding annotation files (.txt) 
- - A `.yaml`file containing information about our classes and folder location -> an example file can be found here: [data.yaml](data.yaml)

<br><br><br>

### 1.2 Download an annotated Training Dataset and train it on your machine

Sources:

[Huggingface](huggingface.co)
[Roboflow](universe.roboflow.com)
[kaggle](kaggle.com)

A not very sophisticated example dataset:
[Open/Close Eyes Dataset](https://universe.roboflow.com/isee-gufmk/eyes-zefum/dataset/6)

<br>

1. Extract downloaded dataset folder into your project folder.
2. In your project folder create a file named `train.py`
3. Code:
    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="yourdatasetfolder/data.yaml", epochs=100, imgsz=640)
    ```
4. Start the training
    ```bash
    python train.py
    ```

5. Wait until finished
6. Navigate to the newly created folder `runs/train/weights`and find `best.pt`
7. Copy best.pt save it to a different location and name it `mytraining.pt`

<br><br><br>

### 1.3 Labeling Tools

Use a program like X-AnyLabeling locally to label your own datasets. This software is open source and completely free:
[X-AnyLabelling Download Page](https://github.com/CVHub520/X-AnyLabeling/releases)

**Mac Users:** If you encounter the error that the App was blocked because it's not from a verified developer: Go to System Settings > Security > Scroll down, find the blocked app window and press `open anyways`
 
Or use online annotation tools, for example [makesense.ai](makesense.ai) or [Roboflow](roboflow.com).

For this class, we will stick with X-AnyLabeling. Be aware that it might take a moment to start. 

<br><br><br>

### 1.4 Train on any object

1. Collect about 50-100 images of your object. Either by taking them at the spot, or by finding them through your Photo Library. Ofc, more is better.

2. Add some photos (about 20% of your dataset) that aren't depicting your face. 

3. Label all of the Photos in X-Anylabelling containing your face with one class, choose a name = your_name

4. Export the dataset with Tools > Export Annotations. Choose YOLO (.txt) format and set an output folder different that the current one.

5. Split the dataset: Once present, X-Anylabeling kind of ditched the folder export for the correct training folder structure. We have to do this seperately; I found this handy python script that does the hard work for us:

[split_yolo_dataset.py](split_yolo_dataset.py)
Download, place in your project folder and run. 

<br><br><br>

### 1.5 (Optional) Upload your dataset to our owncloud and train next time in the Coding Lab

1. Make sure your dataset is in the right format with train and val folders in place and contains a .yaml file.
   
2. Create a folder with your name on our Owncloud [2026W – S05594 – Machine Learning for Artists](https://base.uni-ak.ac.at/cloud/index.php/f/26312563) in `Datasets`

3. Upload your dataset folders.

<br><br><br>


## 2. DeepFace

We have noticed, that a reliable detection of the face (especially in different environments/surroundings/lighting situations) is only possible with a vast amount of data. Even One-Shot systems like YOLOE fail more or less detecting the same face reliably. 

For face detection we can use [Deepface](https://github.com/serengil/deepface) to detect faces only using a single reference image.

<br><br><br>

### 2.1 Setup

1. Create a new folder on your computer, name it accordingly

2. Open the folder in VS Code and create a virtual environment, as we did with YOLO. 

3. In session05 on GitHub navigate to folder `Deepface` and download the [deepface_requirements.txt](deepface_requirements.txt).

4. Place the deepface_requirements.txt file in your folder 

5. Install requirements with: 
    ```bash
    pip install -r deepface_requirements.txt
    pip install deepface
    ```

<br><br><br>
    
### 2.2 Basic Example

For this basic example we try to read out our webcam to detect on faces. The `DeepFace.analyze` can detect (problematically) emotion, age, gender and race. 

1. Create a new python script, call it `basic.py`

2. Import your libraries
    ```python
    from deepface import DeepFace
    import cv2
    ```

3. Initialize your Webcam – a true classic.
    ```python
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        exit()
    ```

4. Start the main Loop and read frames continuously
    ```python
    while True:
        ret, frame = cam.read()
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

    cam.release()
    cv2.destroyAllWindows()
    ```




<details>
<summary>Full Code</summary>

```python
import cv2
from deepface import DeepFace

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cam.read()
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

cam.release()
cv2.destroyAllWindows()

```

</details>

<br><br><br>

### 2.3 Verify Faces

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
        result = DeepFace.verify(reference_image, frame, model_name="Facenet", detector_backend="opencv") # ArcFace, Facenet512, VGG-Face (best), SFace, DeepID, OpenFace...
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
        cv2.rectangle(frame, (x, q), (x + w, y + h), color, 2)
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

## 3. Mediapipe

Mediapipe is a framework by Google for real-time computer vision applications. It's good at looking at your body for use cases like **hand tracking, face mesh, pose estimation, object detection and holistic body tracking**.

These examples use the Tasks API (`mediapipe.tasks`). Each detector is a separate task, and each task needs a small model file ending in `.task`. Put those files in the same folder as your script.

### 3.1 Setup

1. Create a folder on your computer dedicated to mediapipe.

2. Create a virtual environment

3. Install mediapipe via pip. `scipy` is only needed later, for the blink example.

    ```bash
    pip install mediapipe opencv-python numpy scipy
    ```

4. Download the hand and face models into that folder.

    ```bash
    curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
    ```

<br><br><br>

### 3.2 Simple Hand Tracking

1. Import opencv and mediapipe, and open the webcam.
    ```python
    import cv2
    import mediapipe as mp

    cap = cv2.VideoCapture(0)
    ```

2. Pick the pieces we need from the Tasks API. `HAND_CONNECTIONS` is the list of lines between the 21 hand points. Each connection has a `start` and an `end` index. Current MediaPipe has no drawing helper, so this function draws those lines with OpenCV. Landmark positions are 0–1, so multiply by the frame size to get pixels.
    ```python
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    def draw_landmarks(image, landmarks, connections):
        height, width = image.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for connection in connections:
            cv2.line(image, points[connection.start], points[connection.end], (255, 255, 255), 2)
        for point in points:
            cv2.circle(image, point, 4, (0, 0, 255), -1)
    ```

3. Load `hand_landmarker.task` and set the detector to video mode. Video mode tracks the hand from frame to frame, so each call needs a timestamp that keeps going up. `num_hands` is how many hands to look for.
    ```python
    hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    timestamp_ms = 0
    ```

4. Read frames from the webcam.
    ```python
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    ```

5. Flip the image so it feels like a mirror. OpenCV frames are BGR, and MediaPipe wants RGB, so convert the frame and wrap it in an `mp.Image`. `detect_for_video` needs the timestamp from step 3. Then move the clock forward for the next frame.
    ```python
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = hands.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33
    ```

6. `results.hand_landmarks` is a list with one entry per hand. Each entry is the 21 points. Draw them on the original BGR frame.
    ```python
        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)
    ```

7. Show the image, and quit when `q` is pressed.
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

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

def draw_landmarks(image, landmarks, connections):
    height, width = image.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for connection in connections:
        cv2.line(image, points[connection.start], points[connection.end], (255, 255, 255), 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 0, 255), -1)

cap = cv2.VideoCapture(0)
hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))
timestamp_ms = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = hands.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)

    cv2.imshow('Hands', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

### 3.3 Hand Zoom Example

We can calculate the distance between fingers with this code. The 21 landmarks are listed [here.](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) The tip of the thumb is point 4, the tip of the pinky is point 20. In the Tasks API each hand is a list, so those points are `hand_landmarks[4]` and `hand_landmarks[20]`. The `distance` drives the zoom of the image.

1. Add Numpy to the imports. We need it to measure the distance. The MediaPipe names are the same ones from the hand tracking example.
    ```python
    import cv2
    import mediapipe as mp
    import numpy as np

    cap = cv2.VideoCapture(0)

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode
    HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    def draw_landmarks(image, landmarks, connections):
        height, width = image.shape[:2]
        points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
        for connection in connections:
            cv2.line(image, points[connection.start], points[connection.end], (255, 255, 255), 2)
        for point in points:
            cv2.circle(image, point, 4, (0, 0, 255), -1)
    ```

2. This function reads the two fingertips and returns the distance between them in pixels. Landmark coordinates are 0–1, so multiply by the frame size to get pixels. `np.linalg.norm` is the length of the line between the two points.
    ```python
    def get_pinky_thumb_distance(hand_landmarks, frame_width, frame_height):
        thumb_tip = hand_landmarks[4]
        pinky_tip = hand_landmarks[20]
        x1, y1 = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
        x2, y2 = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)
        return np.linalg.norm([x2 - x1, y2 - y1])
    ```

3. Load the hand landmarker and set the starting zoom. `base_distance` is filled in on the first frame where a hand is visible, and later distances are compared to it.
    ```python
    hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))
    zoom = 1.0
    base_distance = None
    timestamp_ms = 0
    ```

4. Read and flip the frame, then run the detector. Same RGB conversion and rising timestamp as in the hand tracking example.
    ```python
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = hands.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33
    ```

5. Get the dimensions of the video frame.
    ```python
        h, w = frame.shape[:2]
    ```

6. If a hand was found, take the first one and draw its connections.
    ```python
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)
    ```

7. Call the function from step 2 to measure the distance between the fingertips.
    ```python
            distance = get_pinky_thumb_distance(hand_landmarks, w, h)
    ```

8. Store the first distance as the baseline. Dividing the current distance by that baseline gives the zoom. `np.clip` keeps it between 1× and 3×.
    ```python
            if base_distance is None:
                base_distance = distance if distance > 0 else 1

            zoom = np.clip(distance / base_distance, 1.0, 3.0)
    ```

9. Draw a line between the fingers and print the zoom value.
    ```python
            thumb_tip = hand_landmarks[4]
            pinky_tip = hand_landmarks[20]
            pt1 = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            pt2 = (int(pinky_tip.x * w), int(pinky_tip.y * h))
            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
            cv2.putText(frame, f"Zoom: {zoom:.2f}x", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    ```

10. When no hand is visible, forget the baseline and reset the zoom.
    ```python
        else:
            base_distance = None
            zoom = 1.0
    ```

11. Crop toward the center of the frame and scale that crop back to the full frame. This only runs when the zoom is above 1.
    ```python
        if zoom > 1.0:
            center_x, center_y = w // 2, h // 2
            new_w, new_h = int(w / zoom), int(h / zoom)
            x1 = max(center_x - new_w // 2, 0)
            y1 = max(center_y - new_h // 2, 0)
            x2 = min(center_x + new_w // 2, w)
            y2 = min(center_y + new_h // 2, h)
            cropped = frame[y1:y2, x1:x2]
            frame = cv2.resize(cropped, (w, h))
    ```

12. Show the image and quit when `q` is pressed.
    ```python
        cv2.imshow('MediaPipe Hands Zoom', frame)
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
import numpy as np

cap = cv2.VideoCapture(0)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HAND_CONNECTIONS = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

def draw_landmarks(image, landmarks, connections):
    height, width = image.shape[:2]
    points = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks]
    for connection in connections:
        cv2.line(image, points[connection.start], points[connection.end], (255, 255, 255), 2)
    for point in points:
        cv2.circle(image, point, 4, (0, 0, 255), -1)

def get_pinky_thumb_distance(hand_landmarks, frame_width, frame_height):
    thumb_tip = hand_landmarks[4]
    pinky_tip = hand_landmarks[20]
    x1, y1 = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
    x2, y2 = int(pinky_tip.x * frame_width), int(pinky_tip.y * frame_height)
    return np.linalg.norm([x2 - x1, y2 - y1])

hands = HandLandmarker.create_from_options(HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))
zoom = 1.0
base_distance = None
timestamp_ms = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = hands.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33

    h, w = frame.shape[:2]

    if results.hand_landmarks:
        hand_landmarks = results.hand_landmarks[0]
        draw_landmarks(frame, hand_landmarks, HAND_CONNECTIONS)

        distance = get_pinky_thumb_distance(hand_landmarks, w, h)
        if base_distance is None:
            base_distance = distance if distance > 0 else 1

        zoom = np.clip(distance / base_distance, 1.0, 3.0)

        thumb_tip = hand_landmarks[4]
        pinky_tip = hand_landmarks[20]
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

hands.close()
cap.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>

### 3.4 Blink Recognition

Using the Face Landmarker, we can find out if a person blinks. This works by measuring the eye landmarks and calculating the eye aspect ratio (EAR). The landmark map is [here.](https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png) The Tasks model uses the same point numbers as that map.

1. Import libraries. `scipy` calculates the distances between eyelid points.
    ```python
    import cv2
    import numpy as np
    import mediapipe as mp
    from scipy.spatial import distance
    ```

2. These are the six landmark numbers for each eye: two corners, two points on the upper lid, two on the lower lid.
    ```python
    LEFT_EYE = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE = [362, 385, 387, 263, 373, 380]
    ```

3. Eye Aspect Ratio (EAR). A and B are the vertical distances between the upper and lower lids. C is the horizontal distance between the corners. `ear` is high when the eyes are open and drops when they close.
    ```python
    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear
    ```

4. Set the blink counters. `EAR_THRESHOLD`: below this, the eye is treated as closed. `CONSEC_FRAMES`: a blink counts only if the eye stays closed for at least this many frames.
    ```python
    EAR_THRESHOLD = 0.25
    CONSEC_FRAMES = 2
    blink_count = 0
    frame_counter = 0
    ```

5. Load `face_landmarker.task` in video mode, and open the webcam. `num_faces` is how many faces to look for.
    ```python
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='face_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    ))

    cap = cv2.VideoCapture(0)
    timestamp_ms = 0
    ```

6. Read a frame, convert it to RGB, and run the face landmarker. Then read the frame size.
    ```python
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
        timestamp_ms += 33
        h, w = frame.shape[:2]
    ```

7. `result.face_landmarks` is a list with one entry per face. Each entry is the full list of face points. Take the first face.
    ```python
        if result.face_landmarks:
            mesh_points = result.face_landmarks[0]
    ```

8. Turn the six eye points from 0–1 coordinates into pixel positions.
    ```python
            left_eye = [np.array([mesh_points[i].x * w, mesh_points[i].y * h]) for i in LEFT_EYE]
            right_eye = [np.array([mesh_points[i].x * w, mesh_points[i].y * h]) for i in RIGHT_EYE]
    ```

9. Measure both eyes and average them, so one eye blinking still moves the value.
    ```python
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
    ```

10. Draw a dot on each of those eye landmarks.
    ```python
            for pt in left_eye + right_eye:
                cv2.circle(frame, tuple(np.int32(pt)), 2, (0, 255, 0), -1)
    ```

11. While the EAR stays under the threshold, count frames. When the eye opens again, count one blink if it was closed long enough, then reset the counter.
    ```python
            if ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= CONSEC_FRAMES:
                    blink_count += 1
                frame_counter = 0
    ```

12. Draw the blink count and the current EAR on the frame.
    ```python
            cv2.putText(frame, f"Blinks: {blink_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    ```

13. Show the image and quit when `q` is pressed.
    ```python
        cv2.imshow("Blink Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    face_landmarker.close()
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

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

face_landmarker = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
))

cap = cv2.VideoCapture(0)
timestamp_ms = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_image, timestamp_ms)
    timestamp_ms += 33
    h, w = frame.shape[:2]

    if result.face_landmarks:
        mesh_points = result.face_landmarks[0]
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

face_landmarker.close()
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




## 4. Speech Recognition with Whisper

Whisper is a speech to text model that can be used i.e. for live subtitles

### Setup

1. Create a new folder and open it in VS Code

2. Create a virtual environment

3. Install necessary libraries
    ```bash
    pip install openai-whisper sounddevice numpy opencv-python faster-whisper
    ```

<br><br><br>

### 4.1 Basic Example

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

## 4.2 Live Subtitles

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


## 4.3 Recognize Trigger Words

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

<br><br><br>


## 5. Getting the data out of the box with OSC
Open Sound Control (OSC) is a protocol for sending simple data strings over the network. It is super easy to use and supported in a wide range of programs/languages. You can either send messages locally on your device, or send them to other machines. The devices have to be on the same network though.

Before starting, make sure to install python-osc with:
```bash
pip install python-osc
```

For easier monitoring of incoming OSC Values "Protokol" by Hexler is recommended. There is also an iOS/Android App which makes testing out the network connection super easy. [Download Protokol](https://hexler.net/protokol#get)

<br><br><br>

### Simple Sender Script

1. Import necessary libraries
    ```python
    from pythonosc import udp_client
    import time
    ```

2. Define which at IP address you want to send the message to:
   1. Local (Your computer)
        ```python
        ip = "127.0.0.1"
        ```
    1. Everone on the network
        ```python
        ip = "255.255.255.255"
        ```
    2. Find out the specific ip address of the receiving device
        ```python
        ip = "xxx.xxx.xxx.xxx"
        ```
3. Define the port on both devices
    ```python
    port = 5005
    ```

4. Set up the Client
   ```python
   client = udp_client.SimpleUDPClient(ip, port)
   ```
5. Set and send the message
    ```python
    message = "Hello from PythonOSC!"
    client.send_message("/test", message)

    print(f"Sent: {message}")
    ```

### Simple Receiver Script

```python
from pythonosc import dispatcher
from pythonosc import osc_server

def handle_message(address, *args):
    print(f"Received message at {address}: {args[0]}")

# Set up dispatcher
disp = dispatcher.Dispatcher()
disp.map("/test", handle_message)


ip = "127.0.0.1"  
port = 5005       # must match sender
server = osc_server.BlockingOSCUDPServer((ip, port), disp)

print(f"Listening on {ip}:{port}")
server.serve_forever()
```

### Sending out data from YOLO11 (With the Flamish Scrollers)

1. Modify the python script to send messages, whenever the classes 'person' and 'cell phone' are detected. 

2. Make the script run in fullscreen with OpenCV.



<br><br><br>

