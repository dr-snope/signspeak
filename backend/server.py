import base64
import time
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.download_model import download_model
from ml.predict import predict_sign
from ml.utilities import extract_landmarks

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=download_model()),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Frontend connected to WebSocket")
    
    try:
        while True:
            # 1. Receive base64 frame from frontend
            payload = await websocket.receive_text()
            if "," in payload:
                payload = payload.split(",")[1]

            img_bytes = base64.b64decode(payload)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp = time.monotonic_ns() // 1_000_000
            result = detector.detect_for_video(mp_image, timestamp)

            raw_coords = extract_landmarks(result)
            label, confidence = predict_sign(raw_coords)

            dots = []
            if result.hand_landmarks and len(result.hand_landmarks) > 0:
                for pt in result.hand_landmarks[0]:
                    dots.append({"x": 1.0 - pt.x, "y": pt.y})

            await websocket.send_json({
                "detected": label is not None,
                "sign": label if label else "",
                "confidence": round(confidence * 100, 1),
                "landmarks": dots
            })

    except WebSocketDisconnect:
        print("[INFO] Frontend disconnected from WebSocket")