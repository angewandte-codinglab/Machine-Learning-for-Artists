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
        #print(results[0].boxes)
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
