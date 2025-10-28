"""
Drone with YOLOv8 Integration:
The Tello drone flies in a square path while the YOLOv8 model detects workers, helmets, and vests in real-time.
Bounding boxes are displayed using OpenCV during the flight.
"""

from djitellopy import Tello
from ultralytics import YOLO
import cv2
import time

# Load the trained YOLOv8 model by filename only
model = YOLO("yolo_drone.pt")

# Connect to Tello drone
tello = Tello()
tello.connect() 
print(f"Connected to Tello - Battery: {tello.get_battery()}%")

tello.takeoff()
time.sleep(2) # Small delay

# Move up before starting the square
tello.move_up(50)
print("Drone moved up 50 cm")

# Fly in a square path and run YOLO detection
for direction in ["forward", "right", "back", "left"]:
    if direction == "forward":
        tello.move_forward(100)
    elif direction == "right":
        tello.move_right(100)
    elif direction == "back":
        tello.move_back(100)
    elif direction == "left":
        tello.move_left(100)
    
    print(f"Moved {direction} 100 cm")
    time.sleep(1)

    # Capture frame from drone camera
    frame = tello.get_frame_read().frame
    # Run YOLO inference
    results = model(frame)
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    # Show frame
    cv2.imshow("Drone YOLO Detection", annotated_frame)
    cv2.waitKey(1)  # Small delay to render frame

# Rotate drone
tello.rotate_clockwise(360)
print("Drone rotated 360 degrees")

# Landing
tello.land()
cv2.destroyAllWindows()