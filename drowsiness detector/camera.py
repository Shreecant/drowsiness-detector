import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
else:
    ret, frame = cap.read()
    if ret:
        print("Camera is working!")
    else:
        print("Camera opened but couldn't read frame")
cap.release()
