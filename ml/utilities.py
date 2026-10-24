"""
SignSpeak - ML Utilities
Shared mathematical functions and data loaders.
"""
import numpy as np
import os

# Modules to use:
# - numpy: Array math, reshaping, broadcasting, saving/loading .npy files.
# - os: Creating dataset folders and navigating paths.

# What will be built here:
# 1. extract_landmarks(results): Converts MediaPipe's 21 landmarks into a 63-element array.
# 2. normalize_landmarks(landmarks): Subtracts wrist position (x0, y0, z0) so coordinates are position-independent.
# 3. load_dataset(data_dir): Reads all recorded .npy files and packages them into training arrays (X, y).