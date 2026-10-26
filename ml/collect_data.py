import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utilities import extract_landmarks, normalize_landmarks
from backend.camera import get_frames

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "gestures.csv")
MODEL_PATH = os.path.join("models", "hand_landmarker.task")


def initialize_csv():
    header = ["label"]
    for i in range(21):
        for axis in ["x", "y", "z"]:
            header.append(f"{axis}{i}")

    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_sample(label, normalized_coords):
    row = [label] + list(normalized_coords)
    with open(CSV_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def collect_gestures():
    initialize_csv()

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] '{MODEL_PATH}' not found. Please ensure hand_landmarker.task exists.")
        return

    sign_label = input("Enter the sign label to record (e.g. HELLO): ").strip().upper()
    if not sign_label:
        print("[ERROR] Label cannot be empty.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1
    )
    detector = vision.HandLandmarker.create_from_options(options)

    count = 0
    print(f"\n[STARTED] Recording for '{sign_label}'")
    print("Controls: Press [SPACE] to capture snapshot | Press [Q] to quit\n")

    for frame in get_frames():
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)
        if result.hand_landmarks:
            for joint in result.hand_landmarks[0]:
                cx, cy = int(joint.x * w), int(joint.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"Label: {sign_label} | Samples: {count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press [SPACE] to capture | [Q] to quit", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("SignSpeak - Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            raw_coords = extract_landmarks(result)
            if not np.allclose(raw_coords, 0):
                normalized = normalize_landmarks(raw_coords)
                append_sample(sign_label, normalized)
                count += 1
                print(f"[SAVED] Sample #{count} for '{sign_label}'")
            else:
                print("[WARNING] No hand detected in frame! Hold your hand up and try again.")

        elif key == ord('q'):
            break

    detector.close()
    cv2.destroyAllWindows()
    print(f"\n[FINISHED] Recorded {count} samples for '{sign_label}' into {CSV_PATH}")


if __name__ == "__main__":
    collect_gestures()