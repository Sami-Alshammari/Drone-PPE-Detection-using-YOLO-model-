"""
This project aims to train a YOLOv8 model that detects workers and ensures
they are following safety procedures by wearing helmets and safety vests.
This script represents the MODEL part of the project, which focuses on training
and evaluating the detection model.
The second part of the project is about controlling a Tello Talent drone using python.
"""

from ultralytics import YOLO

# Load the pre-trained YOLOv8n model
model = YOLO("yolov8n.pt")  # You already have this file in the folder

# Train the model (optional, comment out if you just want to use pre-trained)
data_yaml = r"C:\Users\The OWL\Downloads\drone-data\data.yaml"
results = model.train(
    data=data_yaml,
    epochs=20,
    batch=8,
    imgsz=640,
    project=".",
    name="yolo_drone",
    exist_ok=True,
    device=0  # GPU
)

# Save the  model 
model.save("yolo_drone.pt")


# Validate the model and calculate metrics
metrics = model.val()

# Average metrics
avg_precision, avg_recall, avg_map50, avg_map = metrics.mean_results()
print("\n======")
print(f"Avg Precision = {avg_precision:.4f}")
print(f"Avg Recall    = {avg_recall:.4f}")
print(f"Avg mAP@0.5   = {avg_map50:.4f}")
print(f"Avg mAP@0.5:0.95 = {avg_map:.4f}")

# Detailed metrics for each class
class_summary = metrics.summary()
for cls in class_summary:
    print(f"Class {cls['name']}: Precision={cls['p']:.4f}, Recall={cls['r']:.4f}, F1={cls['f1']:.4f}, mAP50={cls['AP50']:.4f}")
