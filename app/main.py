from fastapi import FastAPI, File, UploadFile
import shutil
import os
from utils import detect_and_crop_objects

app = FastAPI()

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "output"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"

    # Save the uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the image
    cropped_images = detect_and_crop_objects(file_path, OUTPUT_FOLDER)

    return {"message": "Objects detected", "cropped_images": cropped_images}
