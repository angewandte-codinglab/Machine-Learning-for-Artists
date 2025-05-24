import cv2
from ultralytics import YOLO
from pythonosc.udp_client import SimpleUDPClient
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)

model = YOLO('yolo11n.pt')

confidence_threshold = 0.1

osc_ip = "127.0.0.1" 
osc_port = 8000
osc_client = SimpleUDPClient(osc_ip, osc_port)


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

  
    results = model(frame, conf=confidence_threshold, classes=[0, 67])

   
    detected_classes = set()
    if results and results[0].boxes is not None:
        for box in results[0].boxes.data:
            cls = int(box[5])  
            detected_classes.add(cls)

  
    if 0 in detected_classes and 67 in detected_classes:
        print("Stay focused!")
     
        osc_client.send_message("/focus", 1)
    else:
        osc_client.send_message("/focus", 0)


    annotated_frame = results[0].plot()
    cv2.imshow("YOLO11 Detection", annotated_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
