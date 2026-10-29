# Session 04

## 1. Reactivate your virtual environment
If you haven't created one yet, please follow the instructions from [Session 2](../session02/session02.md)

1. Open VS Code and press `CMD + Shift + N` (CTRL + Shift + N on Windows). 

2. Drag your folder into the newly created window.
   
3. Open the Terminal inside VS Code. By default you already should be at the correct folder location.

4. Activate the virtual environment
   ```bash
   source ./venv/bin/activate
   ```
   on Windows
   ```bash
   .\venv\Scripts\activate
   ```

<br><br><br>


## 2. Installing YOLO (Yes it is called like that)


1. Install Ultralytics
   
   ```bash
   pip install ultralytics
   ```

   This should load a lot of dependencies aswell.


<br><br>

## 2.1 Run inference on webcam

Create a new file and call it `webcam.py`

<br>

### webcam.py code breakdown:

Import necessary dependencies:
```python
import cv2
from ultralytics import YOLO
```

Specify which YOLO model you want to use. This points to a "weights"-file (.pt) in your folder and can be interchanged with other weights.
```python
model = YOLO('yolo11n.pt')  
```

We want to use our webcam as a source:
```python
cam = cv2.VideoCapture(0) 
```

Read from the Webcam and print an error if the read fails:
```python
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break
```

Get results from the YOLO inference:
```python
    results = model(frame)
```

Plot the results so we can see them:
```python
    annotated_frame = results[0].plot()
```

Show the results in a new window:
```python
    cv2.imshow("Yolo11 Webcam", annotated_frame)
```

Break the loop and close the program:
```python
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

<br>


<details>
<summary>Full Code</summary>

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

**On my installation i ran into an error with numpy, that wouldn't let me start the script. If that's the case, you need to install an older version of numpy: `pip install "numpy<2"**

<br><br>


## 2.2 Inference on Video

Specify the path to the Video File and run the detector.

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

video_file = "videos/Barcelona opera reopens with performance for more than 2000 potted plants.mp4"

results = model(video_file, save=True, conf=0.25)

print("done")
```

<br><br>

## 2.3. Inference on images

### Breakdown of the code:

These are our imports – additionally to ultralytics we will import **pathlib** which gives us access to the folders on our computer. We don't need opencv here because we are not showing the inference live. 

You might need to install pathlib by executing `pip install pathlib`

```python
from ultralytics import YOLO
from pathlib import Path
```

Reference your YOLO-Model
```python
model = YOLO("yolo11n.pt") 
```

Specify the path to your folder containing the images you want to run the detector on
```python
root_dir = Path("images/")
```

Now we need to make an array list out of the file names to we can iterate on each image.

`str(p)` = function to make a string out of the path

`.glob` = searches through all the file in the folder – use `.rglob` to search all the subfolders too

```python
image_files = [str(p) for p in root_dir.glob("*") if p.is_file()]
```

Next we need a loop that runs the prediction on each file
```python
for image_file in image_files:
    print(f"Processing {image_file}")
    results = model(image_file, save_crop=False, save=True, save_txt=False, conf=0.4)
```
This should create a folder structure like `runs/detect/predict` where all your annotated images will be saved.


<br><br>

## 2.4 Run inference on a stream from inseccam.org
Modify your webcam script for this. It can stay mostly the same, but we'll need to use another source. Some streams don't work – they might require a login. Just choose another stream.

1. Get your videostream from insecam.org. Rightclick on a video stream and press 'Copy Image Address'
   
2. Create a variable after your imports
   ```python
   #Paste your URL
   stream_url = 'http://url.mjpg'
   ```

3. Change this line below:
   ```python
   cam = cv2.VideoCapture(stream_url) 
   ```

<details>
<summary>Full Code</summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

stream_url = 'http://190.210.250.149:91/mjpg/video.mjpg'

cam = cv2.VideoCapture(stream_url) 

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to get Stream.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO11 Insecam.org", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```
</details>

<br><br>

## 2.5 Inference on Twitch Stream
To load a Twitch (or YouTube) videostream, we need to install another library first called `streamlink`

1. Install streamlink and ffmpeg-python (for video encoding). It's possible to install multiple libraries just by separating them with space
   ```bash
   pip install ffmpeg-python streamlink
   ```
2. In the python script, we need to change the imports. Add streamlink and subprocess.
   ```python
    import subprocess
    import streamlink
    ```
3. Import CV2, YOLO and define model
   ```python
   import cv2
   from ultralytics import YOLO

   model = YOLO("yolo11n.pt")
   ```
   
4. To correctly implement the stream, we need to write a short function.
   
   1. Define twitch url
        ```python
        TWITCH_URL = 'your_twitch_url
        ```
    2. Get Stream URL function
        ```python
        def get_stream_url(url):
            streams = streamlink.streams(url)
            if 'best' in streams:
                return streams['best'].url
            else:
                raise Exception("Could not retrieve stream.")
        ```
5. Let's put our inference also into function, so we can call it later in a main function.
    ```python
    def run_inference():
        stream_url = get_stream_url(TWITCH_URL)

        # Open video stream with OpenCV
        cam = cv2.VideoCapture(stream_url)
        if not cam.isOpened():
            print("Failed to open Twitch stream.")
            return

        print("Streaming and running inference... Press 'q' to quit.")

        while True:
            ret, frame = cam.read()
            if not ret:
                print("Stream ended or frame couldn't be read.")
                break

            # Run inference
            results = model(frame)

            # Draw results on frame
            annotated_frame = results[0].plot()

            # Show frame
            cv2.imshow("YOLO11 on Twitch Stream", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cam.release()
    cv2.destroyAllWindows()
    ```

6. Call the inference function in a main function:
    ```python
    if __name__ == '__main__':
    run_inference()
    ```

<br>

<details>
<summary>Full Code</summary>
   
```python
import cv2
import subprocess
from ultralytics import YOLO
import streamlink

TWITCH_URL = ''

model = YOLO('yolo11n.pt')

def get_stream_url(url):
    
    streams = streamlink.streams(url)
    if 'best' in streams:
        return streams['best'].url
    else:
        raise Exception("Could not retrieve stream.")

def run_inference():
    stream_url = get_stream_url(TWITCH_URL)

    cam = cv2.VideoCapture(stream_url)
    if not cam.isOpened():
        print("Failed to open Twitch stream.")
        return

    print("Streaming and running inference... Press 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Stream ended or frame couldn't be read.")
            break

        results = model(frame)

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO11 on Twitch Stream", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()

```
</details>

<br><br>

## 2.6 Inference on Youtube Stream

Same as above, just change `TWITCH_URL` to `YOUTUBE_URL` and add your youtube url.

<br><br><br>


## 3. Finetuning

**Confidence Threshold**

You can set the confidence threshold to a value between 0 and 1. Detections below will not be shown. Change this line:

```python
results = model(frame, conf=0.5)
```

**IOU Threshold**

Specifies how much the detections can overlap – this is used to eliminate multiple detections on the same object to get a clear output
```python
results = model(frame, iou=0.5)
```

**Image Size**

Sets how the size of the image that will be shown to the detector. Larger images result in slower detections.
```python
results = model(frame, imgsz=1280)
```


**Device**

If you have a graphics card (NVIDIA or Apple M1/2/3/4 Chip) you can run the inference on it. This drastically improves the speed.
```python
#CPU
results = model(frame, device=cpu)
#GPU
results = model(frame, device=0)
# Apple Silicon (M1 Chip and above)
results = model(frame, device="mps")
```


**Maximum Number of Detections**

Sets the max number of detection. Useful if you just want to detect 1 object.
```python
results = model(frame, max_det=1)
```

 

**Classes**

Filters predictions to a set of class IDs. Only detections belonging to the specified classes will be returned. Useful for focusing on relevant objects in multi-class detection tasks.
```python
results = model(frame, classes=[0])
```

For YOLO11n the classes are listed [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

<br><br>


## 4. The Flemish Scrollers

A simplified reconstruction of Dries Depoorters work. We'll modify our Python script to print a message when both the classes `person` and `cell phone` are being detected.

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

<br><br><br>

1. **Find out the number of the classes we want to detect**

    For YOLO11n the classes are listed [here](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco.yaml)

    <details>

    <summary>Classes</summary>

    `person: 0`
    `cell phone: 67`

    </details>


2. **Set the detector to only detect these two classes**

    Change this line:

    ```python
    results = model(frame, classes=[0, 67])
    ```

3. **Unclutter your print**

    The YOLO detector should now only detect the set two classes. Now we want to get a message in the print if both classes are seen. But YOLO is already printing lots of messages – let's suppress those first

    Change this line:

    ```python
    results = model(frame, classes=[0, 67], verbose=False)
    ```

4. **Find the relevant data**

    We now we want to find where to talk to the yolo detector. We can try to find the relevant strings by just printing out to the console.

    ```python
    print(results)
    ```
    narrowing down our search we find out that results is actually an array list so we need to adress it like this `results[0]

    ```python
    print(results[0].boxes)
    ```
    on a further look we find the `cls` value (=class). Lets print this then:

    ```python
    print(results[0].boxes.cls)
    ````


5. **Put the data in the right place (another list)**



    after the line with `results` add this:

    ```python
    detected_classes = set()
    if results[0].boxes is not None:
        for item in results[0].boxes.cls:
            detected_classes.add(int(item))
    ```

    This creates the variable `detected_classes`. If results from the detector are coming in, we look for the fifth value of the detection box – which is the class number – and stores it to `detected_classes`

    If you want to know how a for loop works check [this](https://www.w3schools.com/python/python_for_loops.asp) for reference.

6. **Print if both classes are detected**

    This condition triggers the print if both classes `0` and `67` are detected.

    ```python
    if both 0 in detected_classes and 67 in detected_classes:
        print("Stay focused!")
    ```

7. **Run on YouTube Stream**
   
   Combine the Inference on YouTube Script with the script we created above!
<br>



<details>

<summary>Full Code</summary>

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

<details>

<summary>Full Code with Counter</summary>

```python
import cv2
import time
from ultralytics import YOLO

model = YOLO('yolo11n.pt')  

confidence_threshold = 0.1

cam = cv2.VideoCapture(0) 
both_visible_since = None
message_printed = False

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
        if both_visible_since is None:
            both_visible_since = time.monotonic()
        visible_duration = time.monotonic() - both_visible_since
        if visible_duration > 3:
            if not message_printed:
                print("\rstay focused (3.0 seconds)          ")
                message_printed = True
        else:
            print(f"\rVisible together: {visible_duration:.1f}/3.0 seconds", end="", flush=True)
    else:
        if both_visible_since is not None:
            print("\r" + " " * 50 + "\r", end="", flush=True)
        both_visible_since = None
        message_printed = False
    
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>


<br><br>




## 5. Using different models

Lets try out some other YOLO models from Ultralytics. They will be downloaded automatically when you start the program.

### Official YOLO-Weights from Ultralytics

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

<br><br>


## 6. More Yolo Applications

## 6.1 Blur Boxes (easy)

By default, blurring a detection box is supported by Ultralytics. You can find more solutions like cropping, counting etc. [here](https://docs.ultralytics.com/solutions/)

1. Change the import to import the solutions
```python
import cv2

from ultralytics import solutions

cam = cv2.VideoCapture(0)
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
while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Apply object blurring
    results = blurrer(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # print(results)  # access the output


cam.release()

cv2.destroyAllWindows()  # destroy all opened windows
```




## 6.1.1 Person Blur with yolo11-seg (intermediate)

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
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Error: Could not open webcam.")
        exit()
    
    while True:
        ret, frame = cam.read()
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

7. YOLO returns one result for this frame. Each detection has a mask and a class id. Keep only people (class `0`), scale each small mask up to the frame, and paint it into the empty mask from step 6. White pixels are the people we will blur.
    ```python
    result = results[0]  # one frame → one result

    if result.masks is not None:
        for person_mask, class_id in zip(result.masks.data, result.boxes.cls):
            if int(class_id) != 0:  # 0 = person
                continue

            # masks are tiny 0/1 images — scale them to 0–255 and to the frame size
            person_mask = person_mask.cpu().numpy()
            person_mask = (person_mask * 255).astype(np.uint8)
            person_mask = cv2.resize(person_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # add this person to the combined mask
            mask = cv2.bitwise_or(mask, person_mask)
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

    cam.release()
    cv2.destroyAllWindows()
    ```


<details>
<summary>Full Code</summary>

```python
import cv2
import numpy as np
from ultralytics import YOLO

model = YOLO("yolo11n-seg.pt")

cam = cv2.VideoCapture(0) 

while True:
    ret, frame = cam.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    results = model(frame, verbose=False)

    mask = np.zeros((height, width), dtype=np.uint8)

    result = results[0]

    if result.masks is not None:
        for person_mask, class_id in zip(result.masks.data, result.boxes.cls):
            if int(class_id) != 0:  # 0 = person
                continue

            # masks are tiny 0/1 images — scale them to 0–255 and to the frame size
            person_mask = person_mask.cpu().numpy()
            person_mask = (person_mask * 255).astype(np.uint8)
            person_mask = cv2.resize(person_mask, (width, height), interpolation=cv2.INTER_NEAREST)

            # add this person to the combined mask
            mask = cv2.bitwise_or(mask, person_mask)

            mask_3ch = cv2.merge([mask, mask, mask])

            blurred = cv2.GaussianBlur(frame, (51, 51), 0)
            result_frame = np.where(mask_3ch == 255, blurred, frame)
            cv2.imshow("Blurred Background", result_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cam.release()
cv2.destroyAllWindows()
```

</details>

<br><br><br>



## 7. DeepFace

We have noticed, that a reliable detection of the face (especially in different environments/surroundings/lighting situations) is only possible with a vast amount of data. Even One-Shot systems like YOLOE fail more or less detecting the same face reliably. 

For face detection we can use [Deepface](https://github.com/serengil/deepface) to detect faces only using a single reference image.

<br><br><br>

### 7.1 Setup

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
    
### 7.2 Basic Example

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
        print("Error: Failed to camture frame.")
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

### 7.3 Verify Faces

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