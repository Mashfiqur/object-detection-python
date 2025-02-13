from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
from detection import detect_and_crop_objects
from search import build_product_index, find_similar_products
from io import BytesIO
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with specific origins like ["http://localhost"] if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Build product index at startup
@app.on_event("startup")
def startup_event():
    build_product_index()
    print("FAISS index built successfully.")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    pil_img = Image.open(BytesIO(file_bytes)).convert("RGB")

    # Process image from memory instead of disk
    cropped_images = detect_and_crop_objects(pil_img)

    return {"cropped_images": cropped_images}

@app.post("/search")
async def search_similar_product(file: UploadFile = File(...)):
    file_bytes = await file.read()
    image = Image.open(BytesIO(file_bytes)).convert("RGB")

    # Find similar products
    results = find_similar_products(image)

    return {"similar_products": results}