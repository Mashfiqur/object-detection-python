import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
import os

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

BASE_URL = "http://localhost:8876"

def detect_and_crop_objects(image_path, output_folder="extracts", conf_threshold=0.25):
    """
    Detects objects in an image and crops them into separate images.
    
    :param image_path: Path to the uploaded image
    :param output_folder: Folder to save cropped images
    :return: List of cropped image data (URLs, class names, accuracy)
    """
    image = cv2.imread(image_path)
    results = model(image, conf=conf_threshold)
    
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = float(box.conf[0])
            if confidence < 0.6:
                continue

            class_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            cropped_img = image[y1:y2, x1:x2]
            
            filename = f"{class_name}_{confidence:.2f}_{int(time.time())}.jpg"
            cropped_path = os.path.join(output_folder, filename)
            Image.fromarray(cropped_img).save(cropped_path)

            # Generate URL
            file_url = f"{BASE_URL}/{cropped_path}"

            detections.append({
                "object": file_url,
                "class": class_name,
                "accuracy": confidence
            })
    
    return detections
