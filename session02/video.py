from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")

video_file = "videos/cctv.mp4"

results = model(video_file, save=True, conf=0.25, show_labels=False, line_width=1)

annotated_frame = results[0].plot()
cv2.imshow("YOLO11 Detection", annotated_frame)

print("done")
