"""
SignSpeak - Data Collection Tool
Interactive Push-to-Sign webcam recorder for capturing gesture sequences.
"""
import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Modules to use:
# - cv2: Open webcam, listen for keypresses (Spacebar to record), display live preview.
# - mediapipe: Track 21 hand landmarks in real-time.
# - numpy: Stack 30 consecutive frames into a (30, 63) gesture tensor.
# - utilities: Import your custom normalization and extraction functions.

# What will be built here:
# 1. Setup camera and MediaPipe Hands.
# 2. Define vocabulary (e.g., ["Hello", "Thank You", "Yes", "No"]).
# 3. While spacebar is held, record 30 frames of normalized landmarks.
# 4. Save the gesture as: 'data/gestures/<word>/sample_<number>.npy'.