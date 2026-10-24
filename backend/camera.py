# test by running camera_test.py

import cv2

def get_frames():
    camera = cv2.VideoCapture(0)
    print("camera opened: ", camera.isOpened())

    try:
        while camera.isOpened():
            ret, frame = camera.read()
            if not ret:
                break
            yield frame
    finally:
        camera.release()
        cv2.destroyAllWindows()

