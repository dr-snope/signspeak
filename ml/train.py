"""
SignSpeak - Model Training Engine
Trains a deep LSTM sequence model on recorded hand gesture datasets.
"""
import numpy as np
import os
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Modules to use:
# - scikit-learn: Split dataset into 80% training / 20% validation.
# - tensorflow / keras: Construct the deep learning sequence network.
# - utilities: Load all recorded .npy files from disk.

# What will be built here:
# 1. Load X (shape: [samples, 30 frames, 63 features]) and y (labels).
# 2. Build LSTM Neural Network:
#    - Input: (30, 63)
#    - LSTM Layer 1 (64 units) -> captures motion over time
#    - LSTM Layer 2 (32 units) -> extracts deep gesture pattern
#    - Dense Output Layer -> Softmax probabilities for each sign word.
# 3. Train for 30-50 epochs.
# 4. Save final trained model as: 'models/signspeak_lstm.keras'.