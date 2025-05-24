# Session 05

## 1. Label your own dataset 

Use a program like AnyLabelling locally to label your own datasets. This software is open source and completely free:
[AnyLabelling Download Page](https://github.com/vietanhdev/anylabeling/releases)

**Mac Users:** If you encounter the error that the App was blocked because it's not from a verified developer: Go to System Settings > Security > Scroll down, find the blocked app window and press `open anyways`
 
Or use online annotation tools, either [Roboflow](roboflow.com) or [CVAT](cvat.ai). Both offer a free plan and additionally have useful features like dataset exports in correct formats.

For this class, we will stick with AnyLabeling. Be aware that it might take a moment to start. 

<br><br><br>

### Train on your face

1. Collect about 50-100 images of your face. Either by taking them at the spot, or by finding them through your Photo Library. Ofc, more is better.

2. Add some photos (about 20% of your dataset) that aren't depicting your face. 

3. Label all of the Photos in Anylabelling containing your face with one class = your_name

4. Export the dataset with Tools > Export Annotations. Choose YOLO (.txt) format and set an output folder. Split your dataset into 80% train and 20% val. (0% test) 

<br><br><br>

## 2. Upload your dataset to our owncloud 

1. Make sure your dataset is in the right format with train and val folders in place and contains a .yaml file.
   
2. Create a folder with your name on our Owncloud `2025S – S05594 – Machine Learning for Artists` in `Datasets_Face`

3. Upload your dataset folders.

<br><br><br>



## 3. Using Zero Shot Systems like YOLOE

The recent YOLOE Model can detect 4585 classes. It is significantly slower, but can be used to detect more meaningful connections in images. You can find a list of all its classes [here](https://docs.ultralytics.com/models/yoloe/#predict-usage:~:text=predefined%20list%20of%204%2C585%20classes) 

There are three modes available:

### Prompt free

Basically our webcam script, just with a different model:

```python
model = YOLO('yoloe-11s-seg-pf.pt')  
```

<details>
<summary>Full Code<summary>

```python
import cv2
from ultralytics import YOLO

model = YOLO('yoloe-11s-seg-pf.pt')  

cam = cv2.VideoCapture(0) 

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame, device='mps')

    annotated_frame = results[0].plot()

    cv2.imshow('YOLOE', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
```

</details>

### Text Prompt

```python
from ultralytics import YOLOE
import cv2

model = YOLOE("yoloe-11l-seg.pt")  # or select yoloe-11s/m-seg.pt for different sizes

names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))

cap = cv2.VideoCapture(0)

print("Press '/' to enter new classes (comma-separated). Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame)
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOE', annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('/'):
        print("Enter new class names, separated by commas (e.g., person,car,dog):")
        cap.release()
        cv2.destroyAllWindows()
        new_classes = input("Classes: ")
        names = [cls.strip() for cls in new_classes.split(",") if cls.strip()]
        if names:
            model.set_classes(names, model.get_text_pe(names))
            print(f"Now searching for: {names}")
        cap = cv2.VideoCapture(0)

cap.release()
cv2.destroyAllWindows()
```

### Visual Prompts

```python
import cv2
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

model = YOLOE("yoloe-11s-seg.pt")

reference_images = [
    "images/booth1.jpg" 
]

visual_prompts_list = []


for reference_image_path in reference_images:
    reference_image = cv2.imread(reference_image_path)
    if reference_image is None:
        print(f"Error: Could not load the reference image {reference_image_path}.")
        continue

    
    print(f"Draw the bounding box on {reference_image_path} and press ENTER or SPACE to confirm.")

    bbox = cv2.selectROI(f"Select Bounding Box - {reference_image_path}", reference_image, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()


    x_min, y_min, width, height = bbox
    x_max = x_min + width
    y_max = y_min + height

    
    visual_prompts_list.append({
        "refer_image": reference_image_path,
        "bboxes": np.array([[x_min, y_min, x_max, y_max]]),
        "cls": np.array([0]),  
    })


cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    
    for visual_prompts in visual_prompts_list:
        results = model.predict(
            frame,
            device="mps",
            refer_image=visual_prompts["refer_image"],  
            visual_prompts={
                "bboxes": visual_prompts["bboxes"],
                "cls": visual_prompts["cls"],
            },
            predictor=YOLOEVPSegPredictor,
        )

        
        annotated_frame = results[0].plot()

    
    cv2.imshow("YOLOE Detection - Specific Person", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()

cv2.destroyAllWindows()
