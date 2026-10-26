from camera import get_frames
import cv2

for frame in get_frames():
    cv2.imshow('camera stream press q to quit', frame) # show camera
    if cv2.waitKey(1)  == ord('q'):
        break

