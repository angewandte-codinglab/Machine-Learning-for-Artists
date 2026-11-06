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

4. Export the dataset with Tools > Export Annotations. Choose YOLO (.txt) format and set an output folder. Split your dataset into 80% train and 20% val. (0% test) 

<br><br><br>

### 1.5 (Optional) Upload your dataset to our owncloud and train next time in the Coding Lab

1. Make sure your dataset is in the right format with train and val folders in place and contains a .yaml file.
   
2. Create a folder with your name on our Owncloud `2025S – S05594 – Machine Learning for Artists` in `Datasets_Face`

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

## 4. Getting the data out of the box with OSC
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



## 6. Download an annotated Training Dataset and train it on your machine

### What does a training set look like?
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
- - A `.yaml`file containing information about our classes and folder location

<br><br><br>

### Download an annotated Training Dataset and train it on your machine

Sources:

[Roboflow](universe.roboflow.com)
[kaggle](kaggle.com)

A not very sophisticated dataset:
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

## 7. Label your own dataset 

Use a program like AnyLabelling locally to label your own datasets. This software is open source and completely free:
[AnyLabelling Download Page](https://github.com/vietanhdev/anylabeling/releases)

Or use online annotation tools, either [Roboflow](roboflow.com) or [CVAT](cvat.ai). Both offer a free plan and additionally have useful features like dataset exports in correct formats.

For this class, we will stick with AnyLabeling. Be aware that it might take a moment to start. 

<br><br><br>

### Train on your face

1. Collect about 50-100 images of your face. Either by taking them at the spot, or by finding them through your Photo Library. Ofc, more is better.

2. Add some photos (about 20% of your dataset) that aren't depicting your face. 

3. Label all of the Photos in Anylabelling containing your face with one class = your_name

4. Export the dataset with Tools > Export Annotations. Choose YOLO (.txt) format and set an output folder. Split your dataset into 80% train and 20% val. (0% test) 

<br><br><br>

## 8. Upload your dataset to our owncloud 

1. Make sure your dataset is in the right format with train and val folders in place and contains a .yaml file.
   
2. Create a folder with your name on our Owncloud `2025S – S05594 – Machine Learning for Artists` in `Datasets_Face`

3. Upload your dataset folders.

<br><br><br>
