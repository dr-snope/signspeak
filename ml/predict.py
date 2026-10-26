"""
SignSpeak - Inference Engine
Exposes predict_sign() for Abel's camera feed and real-time frontend streaming.
"""

def load_model(model_path="models/sign_model.pkl"):
    """
    Loads your trained model from disk into memory using joblib.
    Caches it so it only loads once, not on every frame.
    """
    pass


def predict_sign(raw_coords):
    """
    The main production function for SignSpeak:
    1. Takes 63 raw coordinates from Abel's hand detector.
    2. If coords are all zeros (no hand), returns (None, 0.0).
    3. Normalizes coords using normalize_landmarks().
    4. Passes normalized coords to the loaded model.
    5. Returns (predicted_word, confidence_percentage), e.g. ("HELLO", 0.95).
    """
    pass

