import os
import joblib
import numpy as np
from utilities import normalize_landmarks

MODEL_PATH = os.path.join("models", "sign_model.pkl")
_CACHED_MODEL = None


def load_model(model_path=MODEL_PATH):
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[ERROR] Trained model not found at '{model_path}'. "
                f"Please record gestures and run ml/train.py first."
            )
        print(f"[INFO] Loading model from '{model_path}' into memory...")
        _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def predict_sign(raw_coords: np.ndarray, confidence_threshold: float = 0.65):
    if raw_coords is None or np.allclose(raw_coords, 0):
        return None, 0.0

    model = load_model()

    # Preprocess coordinates into a 1x63 feature row
    normalized_coords = normalize_landmarks(raw_coords)
    feature_vector = normalized_coords.reshape(1, -1)

    # Compute prediction probabilities across all classes
    probabilities = model.predict_proba(feature_vector)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    predicted_label = model.classes_[best_index]

    if confidence < confidence_threshold:
        return None, confidence

    return predicted_label, confidence
