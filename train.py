from ultralytics import YOLO
import shutil

# 1. Pre-trained YOLOv8 model load karein
model = YOLO('yolov8n.pt')

# 2. Training start karein
results = model.train(
    data='/content/parking_dataset/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16
)

# 3. Training mukammal hone ke baad best weights ko Google Drive par copy kar lein
shutil.copy('runs/detect/train/weights/best.pt', '/content/drive/MyDrive/best_parking_model.pt')

print("Training mukammal ho gayi aur model weights Google Drive par safe save ho gaye hain!")