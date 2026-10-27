import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from backend.download_model import download_model
    from backend.camera import get_frames
except ImportError:
    from download_model import download_model
    from camera import get_frames

# Initialize MediaPipe once
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=download_model()),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)
detector = vision.HandLandmarker.create_from_options(options)

def track_hands():
    for frame in get_frames():
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # MediaPipe Video mode detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = time.monotonic_ns() // 1_000_000
        result = detector.detect_for_video(mp_image, timestamp)

        # Draw green tracking dots
        if result.hand_landmarks:
            for joint in result.hand_landmarks[0]:
                cx, cy = int(joint.x * w), int(joint.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        yield frame, result


if __name__ == "__main__":
    for frame, result in track_hands():
        cv2.imshow("SignSpeak - Vision Backend Test", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
