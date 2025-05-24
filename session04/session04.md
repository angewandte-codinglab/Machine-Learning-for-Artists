# Session 04

## 1. Reactivate your virtual environment
If you haven't created one yet, please follow the isntructions from [Session 2](../session02/session02.md)

1. Open VS Code and press `CMD + Shift + N` (CTRL + Shift + N on Windows). 

2. Drag your folder into the newly created window.
   
3. Open the Terminal inside VS Code. By default you already should be at the correct folder location.

4. Activate the virtual environment
   ```bash
   source ./Yolo11/bin/activate
   ```
   on Windows
   ```bash
   .\Yolo11\Scripts\activate
   ```

<br><br><br>


## 2. Using different YOLO datasets

To begin, lets try out some other YOLO models from Ultralytics. They will be downloaded automatically when you start the program.

We will modify our webcam script, if you need the code its here:
<details>
<summary>Code Webcam.py</summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

cam = cv2.VideoCapture(0) 

if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()


while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

### Official YOLO-Weights from Ultralytics

**Larger Datasets**
The 'n' in yolo11n stands for nano. This is the smallest dataset created by Ultralytics. For more classes and precise inference it is possible to use larger Datasets with the endings: `s` `m` `l` `xl`. Note that larger dataset take in more computing power.
```python
model = YOLO('yolo11s.pt')
```

```python
model = YOLO('yolo11l.pt')
```


**Segmentation**
```python
model = YOLO('yolo11n-seg.pt')
```

**Pose Estimation**
```python
model = YOLO('yolo11n-pose.pt')
```

**Classification**
```python
model = YOLO('yolo11n-cls.pt')
```

**Custom Weights**
You can download this one from our Github
```python
model = YOLO('fd_violencedetection_1.pt')
```

<br><br><br>


## 3. The Flamish Scrollers on a Youtube Stream

Task: Modify our Flamish Scrollers Script we created in Session 03 to work with a Youtube Script. Make it work in Full Screen with the OpenCV instructions from Session 02.

<details>
<summary>The Flamish Scrollers Code</summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

confidence_threshold = 0.1

cam = cv2.VideoCapture(0) 

if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame, conf=confidence_threshold, classes=[0, 67], verbose=False)

    #print(results)
    #print(results[0].boxes)
    #print(results[0].boxes.cls)

    detected_classes = set()
    if results[0].boxes is not None:
        for item in results[0].boxes.cls:
            detected_classes.add(int(item))

    
    if 0 in detected_classes and 67 in detected_classes:
        print("Stay focused!")
    
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 Detection", annotated_frame)

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


## 5. More Yolo Applications

## 5.1 Blur Boxes (easy)

By default, blurring a detection box is supported by Ultralytics. You can find more solutions like cropping, counting etc. [here](https://docs.ultralytics.com/solutions/)

1. Change the import to import the solutions
```python
import cv2

from ultralytics import solutions

cap = cv2.VideoCapture(0)
```

2. Set up the blurrer object
```python
blurrer = solutions.ObjectBlurrer(
    show=True,  # display the output
    model="yolo11n.pt",  
    # line_width=2, 
    # classes=[0, 2]
    # blur_ratio=0.5,  # adjust percentage of blur intensity, the value in range 0.1 - 1.0
)
```

3. Run on Video Stream
```python
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty")
        break

    results = blurrer(im0)

    # print(results)  # access the output


cap.release()

cv2.destroyAllWindows()  # destroy all opened windows
```




## 5.1.1 Person Blur with yolo11-seg (intermediate)

Create a blurring effect that only works on persons (or other classes). Modify the webcam script for this.

1. In our imports, we need to add `numpy`, this is a very popular library for array operations. For ease of use we can reference it as `np`
    ```python
    import cv2
    import numpy as np
    from ultralytics import YOLO
    ```

2. We need to use the yolo-seg model to get masks
    ```python
    model = YOLO('yolo11n-seg.pt')
    ```

3. Open the webcam – this stays the same
    ```python
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    ```

4. To create a mask of the correct size, we need to find out the dimensions of our frame
    ```python
    height, width = frame.shape[:2]
    ```

5. Feed the Yolo model
    ```python
    results = model(frame, verbose=False)
    ```

6. Create an empty grayscale mask in the size of the video
    ```python
    mask = np.zeros((height, width), dtype=np.uint8)
    ```

7. This function gets the data of the mask created by yolo and converts feeds into the empty grayscale mask we created before.
    ```python
    for result in results:
        if result.masks is not None:
            for seg, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls) == 0:  # class 0 = person
                    seg_mask = (seg.cpu().numpy() * 255).astype(np.uint8)
                    seg_mask_resized = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, seg_mask_resized)
    ```
    
8. Convert the grayscale mask to a color mask - so we can use it with color frames.
    ```python
    mask_3ch = cv2.merge([mask] * 3)
    ```

9. Blur the whole image. (51, 51) is the kernel size, increase to blur more. It only takes odd numbers though.
    ```python
    blurred = cv2.GaussianBlur(frame, (51, 51), 0)
    ````

10. Create the final output by defining that where the mask is white (=255) the regions should be blurred.
    ```python
    result_frame = np.where(mask_3ch == 255, blurred, frame)
    ```

11. Show the combined result
    ```python
    cv2.imshow('YOLO11 - Blurred Persons', result_frame)
    ```

12. Logic to end the script.
    ```python
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    ```


<details>
<summary>Full Code</summary>

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov11n-seg.pt')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    results = model(frame, verbose=False)

    mask = np.zeros((height, width), dtype=np.uint8)

    for result in results:
        if result.masks is not None:
            for seg, cls in zip(result.masks.data, result.boxes.cls):
                if int(cls) == 0: 
                    seg_mask = (seg.cpu().numpy() * 255).astype(np.uint8)
                
                    seg_mask_resized = cv2.resize(seg_mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    mask = cv2.bitwise_or(mask, seg_mask_resized)

    mask_3ch = cv2.merge([mask] * 3)

    blurred = cv2.GaussianBlur(frame, (101, 101), 0)

    result_frame = np.where(mask_3ch == 255, blurred, frame)

    cv2.imshow('YOLOv8 - Blurred Persons', result_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
```

</details>

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
