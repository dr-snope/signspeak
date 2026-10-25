import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from download_model import download_model
from camera import get_frames

options = vision.HandLandmarkerOptions(
    python.BaseOptions(model_asset_path = download_model()),
    running_mode = vision.RunningMode.VIDEO,
    num_hands=2
)

detector = vision.HandLandmarker.create_from_options(options)

def detect_hands(detector, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) #convert to mediapipe image
    timestamp = time.monotonic_ns() // 1_00_000
    result = detector.detect_for_video(image, timestamp)
    return result


for frame in get_frames():
    result = detect_hands(detector, frame)
    print(len(result.hand_landmarks)) # no. of hands
#TODO: extract landmark coordinates
