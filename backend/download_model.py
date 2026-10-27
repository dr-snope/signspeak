'''
downloads MediaPipe Hand Landmarker
'''
import os
import urllib
from urllib.request import urlretrieve

MODEL_URL =("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
MODEL_PATH = "models/hand_landmarker.task"

def download_model():
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    
    os.makedirs("models", exist_ok=True)
    print("Downloading mediapipe hand landmarker task ..")
    urlretrieve(MODEL_URL, MODEL_PATH) #TODO: error handling
    return MODEL_PATH
