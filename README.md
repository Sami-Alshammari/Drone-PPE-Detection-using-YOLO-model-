# Drone-PPE-Detection-using-YOLO-model

## Project Overview
This project aims to develop an AI-powered drone system that detects workers and ensures their compliance with safety procedures by monitoring **Personal Protective Equipment (PPE)**, specifically **helmets and safety vests**. The system leverages the **YOLOv8 object detection model** to identify workers in real-time from drone-captured video streams.  

The project is divided into two main components:  
1. **Model Training** – Training a YOLOv8 model using a drone dataset annotated with worker PPE information.  
2. **Real-Time Drone Inference** – Deploying the trained model on a **Tello Talent drone** to detect workers and monitor safety compliance in real-world environments.  

## Dataset
The dataset includes images captured by drones, annotated with bounding boxes for:  
- Workers wearing helmets  
- Workers wearing safety vests  
- Workers not wearing PPE  

## Features
- Train a YOLOv8 model for PPE detection  
- Evaluate the model with metrics like Precision, Recall, mAP@0.5, and mAP@0.5:0.95  
- Real-time detection and annotation on drone video feed  
- Alerts for non-compliant workers (without helmet or vest)  

## Requirements
- Python 3.8+  
- Libraries:
  - ultralytics
  - opencv-python
  - djitellopy (Drone programming library)
