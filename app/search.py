import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import faiss
import json
from torchvision import models
from PIL import Image
import requests
import os
from io import BytesIO

# Load a pretrained feature extraction model (MobileNetV2)
model = models.mobilenet_v2(pretrained=True).features
model.eval()

# FAISS Index
index = faiss.IndexFlatL2(1280)  # MobileNetV2 outputs 1280-dimensional features
image_features = []
product_data = []  # Stores product details for retrieval

# Image transformation for feature extraction
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def extract_features(image):
    """Extract features from an image tensor using MobileNetV2."""
    if isinstance(image, str):  # If path is given
        img = Image.open(image).convert("RGB")
    elif isinstance(image, bytes):  # If bytes are given
        img = Image.open(BytesIO(image)).convert("RGB")
    else:
        raise ValueError("Invalid image format")

    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img).mean([2, 3]).numpy()  # Global Average Pooling
    return features

def load_products(file_path="products.json"):
    """Load product data from JSON file."""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)

def build_product_index():
    """Build FAISS index from product images."""
    global index, image_features, product_data
    products = load_products()

    for product in products:
        img_url = product["image_src"]
        
        # Download image
        img_data = requests.get(img_url, stream=True).content

        # Extract features from memory
        features = extract_features(img_data)
        index.add(features)  # Add to FAISS index
        image_features.append(features)
        product_data.append(product)  # Store product info

def find_similar_products(query_img, top_k=5):
    """Find the top K most similar products for the given image."""
    # Convert PIL image to bytes
    img_bytes_io = BytesIO()
    query_img.save(img_bytes_io, format="JPEG")
    query_bytes = img_bytes_io.getvalue()

    query_features = extract_features(query_bytes)

    # Search for similar images
    distances, indices = index.search(query_features, top_k)

    results = []
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        if dist > 400:
            continue
        if idx < len(product_data):
            similarity_score = 1 / (1 + dist)  # Convert distance to similarity (optional)
            results.append({
                "rank": i + 1,
                "product": product_data[idx],
                "distance": float(dist),  # L2 distance (lower is better)
                "confidence": float(similarity_score)  # Optional similarity score (higher is better)
            })

    return results
