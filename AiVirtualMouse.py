import cv2
import numpy as np
import HandTrackingModule as htm
import time

# Library that allows us to move around with mouse
import autopy

widthCamera = 640
heightCamera = 480


# Taking input from camera and setting height and width for the frame
capture = cv2.VideoCapture(0)  # 0: External WebCam and 1 Other
capture.set(3, widthCamera)
capture.set(4, heightCamera)
pTime = 0


# Detect hand marks
detector = htm.handDetector(maxHands=1) # Expecting only 1 hand

while True:
    # 1. Find hand Landmarks
    success, img = capture.read()
    img = detector.findHands(img)

    # Finding positions of the hands.
    lmList, bbox = detector.findPosition(img)


    # 2. Get the tip of the index and middle finger
    # 3. Check which fingers are up
    # 4. Only Index finger up means it is in moving mode

    # Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    # Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
