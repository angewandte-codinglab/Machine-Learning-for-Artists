# Session 02

## 0. Command Line Basics

**Echo** prints in the terminal whatever parameter we pass it.

```bash
echo Rage Against the Machine Learning
```

  


**pwd** stands for print working directory and it prints the "place" or directory we are currently at in the computer.

```bash
pwd
```

  


**ls** presents you the contents of the directory you're currently in. It will present you with both the files and other directories your current directory contains.

```bash
ls
```

  


**cd** is short for Change directory and it will take you from your current directory to another. You can also write cd + space + drag your desired folder to move to that location.

```bash
cd Desktop
```

Move more than one directory:

```bash
cd Desktop/Your_Folder/
```

Tip: Use Tab to autocomplete 

Move up:

```bash
cd ..
```

  


**mkdir** stands for make directory and it will create a new directory for you. You have to pass the command the directory name parameter.

```bash
mkdir myfolder
```

  


**rmdir** stands for Remove directory

```bash
rmdir myfolder
```

  


**touch, ni or echo** allow you to create an empty file in your current directory. As a parameter it needs the file name.

MacOS only:

```bash
touch myfile.txt
```

On Windows Powershell do this instead:

```bash
ni myfile.txt
```

Or on Command Prompt:

```bash
echo > myfile.txt
```

  


**rm** to remove the file:

```bash
rm myfile.txt
```

  


You can clone files from github to your folder.

There might be an error that git is not installed. 
On MacOS it should prompt you to install developer tools, please install them. 
For Windows, please follow this [Link](https://git-scm.com/downloads/win), and install the package.
Then re-run the command.

```bash
git clone https://github.com/fdoblhammer/Machine-Learning-for-Artists.git
```

  
  


## 1. Checking if Python is already installed

To run an Object Detector like YOLO, we need the programming language Python.  
YOLO (via Ultralytics) works with **Python 3.8 or newer**. For this course, aim for **Python 3.10, 3.11, or 3.12**.

You do **not** need to make one Python the global default for your whole computer. Later we will create a *virtual environment* for the project — that is what matters.

  


#### MacOS

1. Open **Terminal**
2. Type `python3 --version`, press Enter
3. Also try `python --version`

  


#### Windows

1. Open **Command Prompt** or **PowerShell**
2. Type `python --version`
3. If that fails, also try `py --version`

  


**What you might see**


| Result                                            | What it means                                 | What to do                                                            |
| ------------------------------------------------- | --------------------------------------------- | --------------------------------------------------------------------- |
| `command not found` / `No such file or directory` | Python is missing (or not on PATH)            | Install Python in the next section                                    |
| Windows opens the Microsoft Store                 | Windows is showing a stub, not a real install | Close the Store and install from python.org below                     |
| `Python 3.10.x`, `3.11.x`, or `3.12.x`            | Good enough for YOLO                          | Skip install → go to **Hello Python**, then the virtual environment   |
| `Python 3.8.x`, `3.9.x`, or `3.13.x`              | Often works, but less tested in class         | Prefer installing 3.11 alongside (do not delete your existing Python) |
| `Python 2.x.x`                                    | Too old                                       | Install 3.11                                                          |


  


After any successful check, also run:

```bash
python -c "import sys; print(sys.executable)"
```

(or `python3 -c "import sys; print(sys.executable)"` on Mac if `python` is unknown)

This prints *which* Python binary your shell is using. If something later feels wrong, this path is useful for debugging — just ask me.

  
  


## 2. Installing Python 3.11 (Only do this if you haven't got the right version)

Install **3.11** if you do not already have 3.10–3.12. It is fine to have several Python versions on one machine.

  


### MacOS

1. Download the [Python 3.11 installer for Mac](https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg)
  (works on both Intel and Apple Silicon Macs)
2. Double-click the downloaded `.pkg` file. If macOS blocks it, **right-click → Open**.
3. Follow the installer. At the end, you can run the **Install Certificates** command if the installer suggests it.
4. **Quit Terminal completely** and open a new window, then check:
  ```bash
    python3 --version
  ```
    You want something like `Python 3.11.x`. If you still see an older version, try:
    Use whichever command shows 3.11 for the next steps (especially when creating the virtual environment).
5. Optional but useful: also check that `pip` is available via that same Python:
  ```bash
    python3 -m pip --version
  ```

  


### Windows

1. Download the [Python 3.11 installer for Windows (64-bit)](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
2. Double-click the downloaded `.exe` file.
3. **IMPORTANT:** on the first screen, check **"Add python.exe to PATH"** (bottom of the window), then choose **Install Now**.
4. Wait until installation finishes.
5. **Close and reopen** Command Prompt / PowerShell, then verify:
  ```bash
    python --version
  ```
    You want `Python 3.11.x`. If `python` still fails, try:
6. Check pip:
  ```bash
    python -m pip --version
  ```

**Common Windows pitfalls**

- If `python` opens the Microsoft Store: the PATH checkbox was missed, or an old Store stub is winning. Re-run the installer, choose **Modify**, enable **Add python.exe to PATH**, or ask me.
- Always open a **new** terminal after installing — old windows keep the old PATH.

  


### Hello Python

Open a new Terminal / Command Prompt / PowerShell and type:

1. `python`
  (on Mac, use `python3` if `python` is unknown)
2. `print("Rage Against the Machine Learning")`
3. `exit()`

If that prints the line, your install is good enough to continue.

  
  


## 3. Creating a virtual environment

There is two ways of working with python: Either with or without an virtual environment. When working with python, in most projects you'll likely need to install *Python Libraries*. These libraries can sometimes interfere with each other, i.e. when you need two different versions of the same library for two different project. For a clean workflow, I recommend to use a Virtual Environment, which can be seen as a glass dome where your python project lives. 

1. Create a dedicated folder for your project on your computer at a location where you can find it again.
2. Open VS Code and press `CMD + Shift + N` (CTRL + Shift + N on Windows).
3. Drag your folder into the newly created window.
4. Open the Terminal inside VS Code. By default you already should be at the correct folder location.
5. Create a virtual environment
  ```bash
    python -m venv Yolo11
  ```
6. Activate the virtual environment
  ```bash
   source ./Yolo11/bin/activate
  ```
   on Windows

In your Command Line Window should now have (Yolo11) prepended. This indicates you are working within the Virtual Environment. If you close your Command Line Window you will need to reactivate the environment with the command above.

  
  


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
  ```

  


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
  if/else
    for loops
    while loops
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
9. If a python script stops working, but doesn't exit itself, press `CRTL + C` to stop it from the CLI

  
  


## 5. OpenCV

OpenCV is a library used mainly for real-time computer vision. We will use it to display images and videos and to do some manipulations on them.

  


### Installation

In your CLI type this an press enter to install opencv and wait until it finishes (takes a moment):

```bash
pip install opencv-python
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

1. Values are in px
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

  


### Read an online Video Stream from [insecam.org](http://insecam.org)

```python
import cv2
```



# Replace with the actual stream URL from [insecam.org](http://insecam.org)

```python
stream_url = '[http://YOUR_STREAM_IP/mjpg/video.mjpg](http://YOUR_STREAM_IP/mjpg/video.mjpg)'

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

