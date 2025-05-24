import cv2
from ultralytics import YOLO

model = YOLO('weights/eyes.pt')  

confidence_threshold = 0.2

cam = cv2.VideoCapture(0) 

if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()


while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    #results = model.track(frame, conf=confidence_threshold)
    results = model(frame, conf=confidence_threshold, classes=[0, 67], verbose=False)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLO11 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
