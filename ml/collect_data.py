import os
import csv
import cv2
import numpy as np
from utilities import extract_landmarks, normalize_landmarks
from backend.tracker import track_hands

DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "gestures.csv")


def initialize_csv():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode="w", newline="") as f:
            header = ["label"] + [f"{axis}{i}" for i in range(21) for axis in ["x", "y", "z"]]
            csv.writer(f).writerow(header)


def append_sample(label, normalized_coords):
    with open(CSV_PATH, mode="a", newline="") as f:
        csv.writer(f).writerow([label] + list(normalized_coords))


def collect_gestures():
    initialize_csv()

    sign_label = input("Enter sign label to record (e.g. HELLO): ").strip().upper()
    if not sign_label:
        print("[ERROR] Label cannot be empty.")
        return

    count = 0
    print(f"[STARTED] Recording for '{sign_label}'")
    print("Controls: Press [SPACE] to capture snapshot | Press [Q] to quit")

    for frame, result in track_hands():
        # Display capture status
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
                print("[WARNING] No hand detected in frame!")

        elif key == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"[FINISHED] Recorded {count} samples for '{sign_label}' into {CSV_PATH}")


if __name__ == "__main__":
    collect_gestures()
