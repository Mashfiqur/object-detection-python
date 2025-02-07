import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# Load YOLOv8 model (downloads automatically if not found)
model = YOLO("yolov8m.pt")

def detect_and_crop_objects(image_path, output_folder="output", conf_threshold=0.25):
    """
    Detects objects in an image and crops them into separate images.
    
    :param image_path: Path to the uploaded image
    :param output_folder: Folder to save cropped images
    :return: List of cropped image paths
    """
    # Load image
    image = cv2.imread(image_path)
    results = model(image, conf=conf_threshold)

    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Get confidence
            confidence = float(box.conf[0])
            # Get class name
            class_name = model.names[int(box.cls[0])]
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            cropped_img = image[y1:y2, x1:x2]
            
            # Save cropped image
            cropped_path = f"{output_folder}/{class_name}_{confidence:.2f}_{int(time.time())}.jpg"
            Image.fromarray(cropped_img).save(cropped_path)
            
            detections.append((cropped_path, class_name, confidence))
            
    return detections
