import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import io
import base64

# Load YOLOv8 model
model = YOLO("yolov8m.pt")

def detect_and_crop_objects(pil_img, conf_threshold=0.6):
    # Convert bytes to a NumPy array
    image = np.array(pil_img)

    # Convert RGB to BGR (OpenCV format)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run YOLO detection
    results = model(image, conf=conf_threshold)
    
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            confidence = float(box.conf[0])

            class_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            cropped_img = image[y1:y2, x1:x2]

            # Convert cropped image to base64
            pil_cropped_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))  # Convert back to RGB
            buffered = io.BytesIO()
            pil_cropped_img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

            detections.append({
                "object": f"data:image/jpeg;base64,{img_base64}",
                "class": class_name,
                "accuracy": confidence
            })
    
    return detections
