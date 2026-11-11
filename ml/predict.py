import os
import sys
import joblib
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from ml.utilities import normalize_landmarks
except ImportError:
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

    normalized_coords = normalize_landmarks(raw_coords)
    feature_vector = normalized_coords.reshape(1, -1)

    probabilities = model.predict_proba(feature_vector)[0]
    best_index = int(np.argmax(probabilities))
    confidence = float(probabilities[best_index])
    predicted_label = model.classes_[best_index]

    if confidence < confidence_threshold:
        return None, confidence

    return predicted_label, confidence

if __name__ == "__main__":
    from backend.tracker import track_hands
    from ml.utilities import extract_landmarks
    import cv2

    print("[INFO] Starting real-time test. Press 'q' to quit.")
    for frame, result in track_hands():
        raw_coords = extract_landmarks(result)
        label, conf = predict_sign(raw_coords)

        display_text = f"Sign: {label} ({conf*100:.1f}%)" if label else "Detecting..."
        cv2.putText(frame, display_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("MudraSetu - Live Prediction Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
