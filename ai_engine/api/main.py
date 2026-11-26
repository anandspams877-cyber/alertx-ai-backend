
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, HTTPException

# ALERTX AI - ONNX Backend

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "fire_smoke_r3_e6_final.onnx"

app = FastAPI(title="ALERTX AI API")

print("Loading ALERTX AI ONNX model...")

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded successfully!")

CLASS_NAMES = ["fire", "smoke"]


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

    original_height, original_width = image.shape[:2]

    # Resize with letterboxing
    size = 320
    ratio = min(size / original_width, size / original_height)

    new_width = int(round(original_width * ratio))
    new_height = int(round(original_height * ratio))

    resized = cv2.resize(image, (new_width, new_height))

    pad_width = size - new_width
    pad_height = size - new_height

    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))
    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114)
    )

    # Prepare ONNX input
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    tensor = rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    # Run inference
    outputs = session.run(None, {input_name: tensor})

    predictions = outputs[0][0].T

    detections = []

    for class_id in range(len(CLASS_NAMES)):

        boxes = []
        scores = []
        coordinates_list = []

        for prediction in predictions:

            confidence = float(prediction[4 + class_id])

            if confidence < 0.25:
                continue

            center_x, center_y, width, height = prediction[:4]

            x1 = float(center_x - width / 2)
            y1 = float(center_y - height / 2)
            x2 = float(center_x + width / 2)
            y2 = float(center_y + height / 2)

            boxes.append([
                x1, y1,
                x2 - x1,
                y2 - y1
            ])

            scores.append(confidence)
            coordinates_list.append([x1, y1, x2, y2])

        if not boxes:
            continue

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            0.25,
            0.45
        )

        if len(indices) == 0:
            continue

        for index in np.array(indices).flatten():

            x1, y1, x2, y2 = coordinates_list[index]

            # Convert coordinates to original image size
            x1 = (x1 - left) / ratio
            y1 = (y1 - top) / ratio
            x2 = (x2 - left) / ratio
            y2 = (y2 - top) / ratio

            x1 = max(0, min(original_width, x1))
            y1 = max(0, min(original_height, y1))
            x2 = max(0, min(original_width, x2))
            y2 = max(0, min(original_height, y2))

            detections.append({
                "class": CLASS_NAMES[class_id],
                "confidence": round(scores[index] * 100, 2),
                "box": [
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2)
                ]
            })

    return {
        "success": True,
        "filename": file.filename,
        "detections_count": len(detections),
        "detections": detections
    }
