# 🅿️ Aerial Parking Space Detection & Free Space Counter using YOLOv8

An automated computer vision solution designed to detect and count available parking spaces in top-view/aerial parking lot images using **YOLOv8**.

## 🎯 Overview
Manual parking management can be inefficient in high-density parking zones. This project leverages deep learning to accurately identify vacant parking slots from aerial imagery, providing real-time counts to streamline vehicle guidance and parking management.

## ✨ Key Features
- **Aerial Object Detection:** Optimized for top-view camera angles and high-density parking lots.
- **Accurate Free Space Counting:** Automatically filters and counts vacant (`free`) parking slots.
- **Bounding Box Optimization:** Uses tuned confidence (`conf=0.6`) and IoU (`iou=0.2`) thresholds to eliminate overlapping detections and noise.
- **Custom Trained:** Trained on annotated aerial parking lot datasets for maximum precision.

## 🛠️ Tech Stack & Tools
- **Framework:** Ultralytics YOLOv8
- **Language:** Python 3.x
- **Computer Vision:** OpenCV, Matplotlib
- **Dataset Format:** YOLO PyTorch / Roboflow

## 📊 Pipeline
1. **Data Preprocessing:** Dataset extraction and annotation structure verification.
2. **Model Training:** Custom YOLOv8 training on bounding box annotations.
3. **Inference & Post-Processing:** Overlap removal and class-based count extraction (`free` slots).
4. **Visualization:** Clean rendered outputs with bounding boxes and counts.
