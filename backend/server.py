import base64
import time

import cv2
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from backend.tracker import load_detector, detect_hand_landmarks
from ml.predict import predict_sign, load_model 
from ml.utilities import extract_landmarks



@asynccontextmanager
async def lifespan(app: FastAPI):
    detector = load_detector()

    try:
        app.state.detector = detector
        app.state.model = load_model()

        print("[INFO] SignSpeak backend ready")
        yield

    finally:
        detector.close()
        print("[INFO] MediaPipe detector closed.")


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[INFO] Frontend connected to WebSocket")
    detector = websocket.app.state.detector

    try:
        while True:
            payload = await websocket.receive_text()
            if "," in payload:
                payload = payload.split(",", 1)[1]

            img_bytes = base64.b64decode(payload)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            #frame = cv2.flip(frame, 1)

            result = detect_hand_landmarks(frame, detector)

            coords = extract_landmarks(result)

            if result.hand_landmarks:
                label, confidence = predict_sign(coords)
            else:
                label, confidence = None, 0.0

            dots = []
            if result.hand_landmarks:
                for pt in result.hand_landmarks[0]:
                    dots.append({
                        "x": pt.x,
                        "y": pt.y,
                    })

            await websocket.send_json({
                "detected": label is not None,
                "sign": label if label else "",
                "confidence": round(confidence * 100, 1),
                "landmarks": dots
            })

    except WebSocketDisconnect:
        print("[INFO] Frontend disconnected from WebSocket")
