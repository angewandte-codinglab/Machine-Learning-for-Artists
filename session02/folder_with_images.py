from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolo11n.pt") 

image_dir = Path("images/")

image_files = [str(p) for p in image_dir.glob("*") if p.is_file()]

# -- Uncomment the next line if you also want to search in subfolders
#image_files = [str(p) for p in root_dir.rglob("*") if p.is_file()]

for image_file in image_files:
    print(f"Processing {image_file}")
    results = model(image_file, save_crop=False, save=True, save_txt=False, conf=0.4)
