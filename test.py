from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

# 1. Google Drive par result save karne ke liye folder set karein
output_dir = '/content/drive/MyDrive/Parking_Test_Results'
os.makedirs(output_dir, exist_ok=True)

# 2. Google Drive se apni trained best weights load karein
model = YOLO('/content/drive/MyDrive/best_parking_model.pt')

# 3. Test image ka path dein
image_path = '/content/parking_dataset/test/images/12-aerial-top-view-parking-lot_51195-161-Copy_jpg.rf.799d8e7c3b2f2146097fdc55506e6d8d.jpg'

# 4. Model run karein (overlap khatam karne ke liye conf aur iou adjust kiye hain)
results = model(image_path, conf=0.5, iou=0.4)

# 5. Free spaces count karne ki logic
free_spaces_count = 0

for r in results:
    boxes = r.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        
        # Check karein ke detected object free space wali class hai ya nahi
        if class_name == 'free':
            free_spaces_count += 1

# Total count print karwayein
print(f"Total Free Parking Spaces: {free_spaces_count}")

# 6. Result image ko display aur Google Drive par save karein
res_plotted = results[0].plot() # BGR format

# Google Drive par image save karne ka path
result_filename = os.path.basename(image_path)
save_path = os.path.join(output_dir, result_filename)
cv2.imwrite(save_path, res_plotted)
print(f"Result image Google Drive par save ho gayi hai: {save_path}")

# Notebook mein image show karne ke liye
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()