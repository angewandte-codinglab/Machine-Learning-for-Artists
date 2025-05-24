## 1. Checking if Python is already installed

To run an Object Detector like YOLO, we need to install the programming language python. Please make sure to follow these steps precicely as it is important to install the correct version with the right configurations.

#### MacOS

1.  Open 'Terminal'
2.  Type `python3 --version`, press Enter
3.  Also try `python --version`

#### Windows

1. Open 'Command Prompt' or 'Powershell'
2. Type `python --version`  

<br>

**This can show 3 different outputs**

1. `No such file or directory` – Python is not installed: You'll need to install python.
2. `Python 3.10.x` or `Python 3.9.x` – Congrats. You already have the right version installed and you can skip the next step. 
3. `Python 2.x.x` or `Python 3.x.x` – You have a different version than 3.10 (or 3.9) installed – You'll need to install 3.10 and tell your machine to use it.
<br><br><br>

## 2. Installing Python 3.10

### MacOS



1. Download the [Python 3.10 Installer for Mac](https://www.python.org/ftp/python/3.10.11/python-3.10.11-macos11.pkg)

2. Double-click the downloaded .pkg File. If you are on MacOS Sonoma or newer, you might get a warning about an unsupported developer app. To bypass this just **right-click** -> **Open**

3. Follow the installer instructions and wait until the installation finishes.

4. Open the Terminal, check if Python3.10 was installed correctly
    ```bash
    python3 --version
    ```

5. Change/Set Python3.10 to be the default Python Version.
    ```bash
    echo 'alias python3="/usr/local/bin/python3.10"' >> ~/.zshrc
    source ~/.zshrc
    ```
6. On MacOS Catalina and later: 
   ```bash
   open -e ~/.zshrc
   ```

   older Macs (bash):
    ```bash
   open -e ~/.zshrc
   ```

7. Add this at the end of the file:
    ```bash
    alias python=python3
    alias pip=pip3
    ```

8. Save and close the file.

9. Apply changes:
    ```bash
   source ~/.zshrc
   ```

   older Macs (bash):
    ```bash
   source ~/.bash_profile
   ```

10. Check it with `python --version`

<br><br>

### Windows

1. Download the [Python3.10 installer for Windows(64bit)](https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe) 
   
2. Double-click the downloaded .pkg File.
   
3. **IMPORTANT:** Check "Add Python to PATH" at the bottom of the Window.
   
4. Install & wait until the installation finishes.
   
5. Verify the installation in Command Prompt/Powershell
   ```bash
   python --version
   ```

    **If this doesnt work, the issue lies mostly within the configuration of the Environment Variables. Here is a [guide](https://readmedium.com/how-to-set-up-a-virtual-environment-with-a-different-python-version-on-windows-10-9900eb0acf9a) on how to set this up correctly, or just ask me.**

<br><br>

### Hello Python

Open Terminal or Command Prompt and type:

1. ```python```
   
2. ```print("Rage Against the Machine Learning")```
   
3. ```exit()```


<br><br><br>

## 3. Creating a virtual environment

There is two ways of working with python: Either with or without an virtual environment. When working with python, in most projects you'll likely need to install *Python Libraries*. These libraries can sometimes interfere with each other, i.e. when you need two different versions of the same library for two different project. For a clean workflow, I recommend to use a Virtual Environment, which can be seen as a glass dome where your python project lives. 

1. Create a dedicated folder for your project on your computer at a location where you can find it again.

2. Open VS Code and press `CMD + Shift + N` (CTRL + Shift + N on Windows). 

3. Drag your folder into the newly created window.
   
4. Open the Terminal inside VS Code. By default you already should be at the correct folder location.

5. Create a virtual environment
    ```bash
    python -m venv Yolo11
    ````
6. Activate the virtual environment
   ```bash
   source ./Yolo11/bin/activate
   ```
   on Windows
   ```bash
   .\Yolo11\Scripts\activate
   ```

In your Command Line Window should now have (Yolo11) prepended. This indicates you are working within the Virtual Environment. If you close your Command Line Window you will need to reactivate the environment with the command above.

<br><br><br>

## 4. Python Basics

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
    ````

<br>

### Python Syntax Basics

Put this into your test.py file

1. **Printing** something to the console works like this
    ```python
    print("Rage Against the Machine Learning)
    ```

2. We use **variables** to store different kind of data
   ```python
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

    The `time` library should come by default with the Python installation, but many libraries that extend Pythons functionality need to be installed first. Make sure your virtual environment is active, then run this in the CLI:
    ```bash
    pip install examplelibrary
    ```

9.  If a python script stops working, but doesn't exit itself, press `CRTL + C` to stop it from the CLI

<br><br><br>

## 5. OpenCV
OpenCV is a library used mainly for real-time computer vision. We will use it to display images and videos and to do some manipulations on them.

### Installation

In your CLI type this an press enter to install opencv and wait until it finishes (takes a moment):
```bash
pip install opencv-python==4.10.0.84
```

At the moment writing, the newest opencv didn't work for me, so we will download version 4.10.0.84

### Load and display in image
1. Find an image on your machine, create a new folder in VS Code called `ìmages` and drag it inside.
2. Create a new python script and name it ?image.py`
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

### Resizing Images
1.  Values are in px
    ```python
    resized_image = cv2.resize(image, (300, 300))

    cv2.imshow('Image', resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

### Full Screen
1. Press any key to exit
    ```python    
    image = cv2.imread('your_image.jpg')

    cv2.namedWindow('FullScreenWindow', cv2.WINDOW_NORMAL)

    cv2.setWindowProperty('FullScreenWindow', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow('FullScreenWindow', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

### Convert the image to Grayscale
1. Some simple image manipulations
   ```python
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Grayscale', gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ```

2. Try out some other manipulations from this table


    | **Action** | **Command** | **Description** |
    |:---|:---|:---|
    | **Grayscale** | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | Convert to black & white. |
    | **Resize** | `cv2.resize(img, (width, height))` | Resize to new dimensions. |
    | **Rotate** | `cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)` | Rotate 90°, 180°, etc. |
    | **Flip** | `cv2.flip(img, flipCode)` | Flip vertically, horizontally, or both (`flipCode = 0, 1, -1`). |
    | **Blur** | `cv2.GaussianBlur(img, (5,5), 0)` | Soft blur (good for smoothing noise). |
    | **Edge Detection** | `cv2.Canny(img, 100, 200)` | Find edges in the image. |
    | **Change Brightness/Contrast** | `img_new = cv2.convertScaleAbs(img, alpha=1.5, beta=50)` | Adjust contrast and brightness. |
    | **Thresholding (Binarization)** | `_, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)` | Convert to pure black and white. |
    | **Draw Text** | `cv2.putText(img, 'Hello', (50,50), font, 1, (255,255,255), 2)` | Write text onto the image. |
    | **Color Space Change** | `cv2.cvtColor(img, cv2.COLOR_BGR2HSV)` | Convert BGR to HSV color space. |
    | **Crop Image** | `cropped = img[y1:y2, x1:x2]` | Slice part of an image (crop). |

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

3.  Show the Video (still in the loop!)
    ```python
        cv2.imshow('Webcam', frame)
    ```

4.  Exit when the key 'q' is pressed
    ```python
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    ```

5. Release the cam and close window (outside loop)
    ```python
    cap.release()
    cv2.destroyAllWindows()
    ```

### Read an online Video Stream from insecam.org
    ```python
    import cv2

    # Replace with the actual stream URL from insecam.org
    stream_url = 'http://YOUR_STREAM_IP/mjpg/video.mjpg'

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

