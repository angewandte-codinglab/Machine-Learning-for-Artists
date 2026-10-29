# Session 03

## 1. Reactivate your virtual environment

If you haven't created a virtual environment yet, please follow the instructions from [Session 2](../session02/session02.md). You should see a `venv` folder in your project folder. VS Studio Code sometimes recognizes your virtual environment automatically, if you see `(venv)` at in you terminal in the beginning of the line you can skip this step as your venv is already active. 

1. Open VS Code and press `CMD + Shift + N` (CTRL + Shift + N on Windows). 

2. Drag your project folder into the newly created window or choose it under `File -> Open Recent...`
   
3. Open the Terminal inside VS Code. By default you already should be at your project's folder location.

4. Activate the virtual environment
   
   ```bash
   source ./venv/bin/activate
   ```
   on Windows
   ```bash
   .\venv\Scripts\activate
   ```


<br><br> 
  


## 2. Python Basics


### Creating and running python script in CLI

1. Lets create a file in VS Code. Press `CMD + N` or `CTRL + N`(Windows)
2. Save this file with the `.py` extension. Name it e.g. `test.py`
3. To run this type into the CLI and press enter:
  ```bash
    python test.py
  ```
4. If your file is in a subfolder you'll need to:
  ```bash
    python foldername/file.py
  ```

<br>

### Python Syntax Basics

Put this into your test.py file

1. **Printing** something to the console works like this
  ```python
    print("Rage Against the Machine Learning)
  ```
2. ```python
   # Numbers
   x = 5
   x = 0.1
   ```
   ```python
   # Strings
   name = "Ferdinand"
   ```
   ```python
   # Booleans
   active = True
   ```
3. **Comments** are used to make your code more readable aswell as to deactivate lines you don't need.
  ```python
   # One Line Comment

   """ 
   Multi 
   Line 
   Comment
   """
  ```
4. **If/else statements** and **loops** 
    <br>if/else
    ```python
    rage = 100
    if rage >= 100:
        print(rage)
    else:
        print("i sleep")
    ```

    for loops
    ```python
    for i in range(500):
        print(i)
    ```

    while loops
    ```python
    rage = 0
    while rage < 100:
        print(rage)
        rage += 5

5. **Functions** make your code reusable
  ```python
    rage = 100

    def learning(rage):
        print(f"Learning: {rage}%")

    learning(rage)
  ```
6. **Lists** can store more than one Value
  ```python
    learning = ["Rage", "against", "the", "Machine"]
    print(learning[1])

    #print(learning[0:4])
  ```
7. **Dictionaries**
  ```python
    progress = {"Rage": "Yes", "Learning": 100}
    print(progress["Rage"])
  ```
8. Working with **libraries**
  ```python
   import time 

   rage = 0

    while rage < 100:
        print(f"{rage} %")
        rage += 5
        time.sleep(1)
  ```
    
  The `time` library should come by default with the Python installation, but many libraries that extend Pythons functionality need to be installed first. To install a library in the venv we use the `pip install` command in the terminal.

  ```bash
  pip install some-library
  ```


9. If a python script stops working, but doesn't exit itself, press `CRTL + C` to stop it from the CLI

  
<br>
  


## 3. OpenCV

OpenCV is a library used mainly for real-time computer vision. We will use it to display images and videos and to do some manipulations on them.

  
<br>

### Installation

In your CLI type this and press enter to install OpenCV and wait until it finishes (takes a moment):

```bash
pip install opencv-python
```

<br>
  
### Load and display in image (working with file paths)

1. Find any image on your machine, create a new folder in VS Code called `ìmages` and drag it inside.
2. Create a new python script and name it `image.py`
3. Import opencv
  ```python
    import cv2
  ```
4. Load image in folder
  ```python
    image = cv2.imread('images/your_image.jpg')
  ```
5. Show image
  ```python
    cv2.imshow('Image Window', image)
  ```
6. Close if any key is pressed
  ```python
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  ```

  
<br>

### Draw Shapes

1. Rectangle
  ```python
    cv2.rectangle(image, (50, 50), (200, 200), (0, 0, 255), 2)
    #Color is in BGR!
  ```
2. Circle
  ```python
    cv2.circle(image, (300, 300), 40, (255, 0, 0), -1)
  ```
3. Results
  ```python
    cv2.imshow('Image Window', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  ```

  
<br>

### Resizing Images

1. Values are in px
  ```python
    resized_image = cv2.resize(image, (300, 300))

    cv2.imshow('Image', resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  ```

<br>


### Full Screen

1. Press any key to exit
  ```python
    image = cv2.imread('your_image.jpg')

    
  ```

  
<br>

### Convert the image to Grayscale

1. Some simple image manipulations
  ```python
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grayscale', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  ```
2. Try out some other manipulations from this table

  | **Action**                      | **Command**                                                        | **Description**                                                 |
  | ------------------------------- | ------------------------------------------------------------------ | --------------------------------------------------------------- |
  | **Grayscale**                   | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`                            | Convert to black & white.                                       |
  | **Resize**                      | `cv2.resize(img, (width, height))`                                 | Resize to new dimensions.                                       |
  | **Rotate**                      | `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)`                         | Rotate 90°, 180°, etc.                                          |
  | **Flip**                        | `cv2.flip(img, flipCode)`                                          | Flip vertically, horizontally, or both (`flipCode = 0, 1, -1`). |
  | **Blur**                        | `cv2.GaussianBlur(img, (5,5), 0)`                                  | Soft blur (good for smoothing noise).                           |
  | **Edge Detection**              | `cv2.Canny(img, 100, 200)`                                         | Find edges in the image.                                        |
  | **Change Brightness/Contrast**  | `img_new = cv2.convertScaleAbs(img, alpha=1.5, beta=50)`           | Adjust contrast and brightness.                                 |
  | **Thresholding (Binarization)** | `_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)` | Convert to pure black and white.                                |
  | **Draw Text**                   | `cv2.putText(img, 'Hello', (50,50), font, 1, (255,255,255), 2)`    | Write text onto the image.                                      |
  | **Color Space Change**          | `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)`                             | Convert BGR to HSV color space.                                 |
  | **Crop Image**                  | `cropped = img[y1:y2, x1:x2]`                                      | Slice part of an image (crop).                                  |


  


### Webcam Video

This chooses the first (0) video device on your computer available. Try to do some image manipulations from above on it.

1. Define the webcam as your source:
  ```python
    cam = cv2.VideoCapture(0) 
  ```
2. Try to read images from the webcam
  ```python
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break
  ```
3. Show the Video (still in the loop!)
  ```python
        cv2.imshow('Webcam', frame)
  ```
4. Exit when the key 'q' is pressed
  ```python
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  ```
5. Release the cam and close window (outside loop)
  ```python
    cap.release()
    cv2.destroyAllWindows()
  ```

  
<br>

### Read an online Video Stream from [insecam.org](https://insecam.org)

Replace with the actual stream URL from [insecam.org](https://insecam.org)

```python
import cv2

stream_url = 'http://89.58.87.210:8081/mjpg/video.mjpg'

cap = cv2.VideoCapture(stream_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to get stream.")
        break

    cv2.imshow('Inseccam.org', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



```

<br>

## 4. Installing YOLO (Yes it is called like that)


1. Install Ultralytics
   
   ```bash
   pip install ultralytics
   ```

   This should load a lot of dependencies aswell.


<br><br>

## 5.1 Run inference on webcam

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


