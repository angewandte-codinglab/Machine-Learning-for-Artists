import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  

cam = cv2.VideoCapture(0) 

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    results = model(frame)

    annotated_frame = results[0].plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
