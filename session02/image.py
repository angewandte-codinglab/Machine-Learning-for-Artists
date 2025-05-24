from ultralytics import YOLO

model = YOLO("yolo11n.pt")

image_file = "images/original_22b598409ede56d4b39e194cad83b495.jpg"

results = model(image_file, save=True, save_txt=True, save_crop=True, conf=0.25)

print("done")
