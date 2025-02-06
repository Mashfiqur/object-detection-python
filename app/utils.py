import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

# Load YOLOv8 model (downloads automatically if not found)
model = YOLO("yolov8n.pt")

def detect_and_crop_objects(image_path, output_folder="output"):
    """
    Detects objects in an image and crops them into separate images.
    
    :param image_path: Path to the uploaded image
    :param output_folder: Folder to save cropped images
    :return: List of cropped image paths
    """
    # Load image
    image = cv2.imread(image_path)
    results = model(image)

    cropped_images = []

    for i, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box.tolist())
        cropped_img = image[y1:y2, x1:x2]

        # Convert cropped image to PIL and save
        cropped_path = f"{output_folder}/object_{i}_{int(time.time())}.jpg"
        Image.fromarray(cropped_img).save(cropped_path)
        cropped_images.append(cropped_path)

    return cropped_images
