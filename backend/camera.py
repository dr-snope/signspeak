import cv2
cam = cv2.VideoCapture(0)
while cam.isOpened():
    ret, frame = cam.read()
    if not ret:
        break
    cv2.imshow('cam feed press q to quit', frame)
    if cv2.waitKey(1) == ord('q'): # keybord input to quit
        break
cam.release()
cv2.destroyAllWindows()
