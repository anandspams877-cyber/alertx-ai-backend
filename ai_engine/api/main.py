
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from ultralytics import YOLO

# ALERTX AI - FastAPI Backend

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "fire_smoke_r3_e6_final.pt"

app = FastAPI(title="ALERTX AI API")

print("Loading ALERTX AI model...")
model = YOLO(str(MODEL_PATH))
print("Model loaded successfully!")

@app.get("/")
def home():
    return {
        "message": "ALERTX AI Backend is running!",
        "status": "online"
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    results = model.predict(
        source=image,
        imgsz=320,
        conf=0.25,
        device="cpu",
        verbose=False
    )

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coordinates = box.xyxy[0].tolist()

            detections.append({
                "class": model.names[class_id],
                "confidence": round(confidence * 100, 2),
                "box": [round(float(value), 2) for value in coordinates]
            })

    return {
        "success": True,
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections
    }
