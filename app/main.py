from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from utils import detect_and_crop_objects

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins like ["http://localhost"] if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

UPLOAD_FOLDER = "uploads"
EXTRACT_FOLDER = "extracts"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(EXTRACT_FOLDER, exist_ok=True)

# Serve extracted images as static files
app.mount("/extracts", StaticFiles(directory=EXTRACT_FOLDER), name="extracts")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = f"{UPLOAD_FOLDER}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    cropped_images = detect_and_crop_objects(file_path, EXTRACT_FOLDER)

    return {"message": "Objects detected", "cropped_images": cropped_images}
