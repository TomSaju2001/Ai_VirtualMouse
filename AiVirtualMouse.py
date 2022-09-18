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


while True:

    success, img = capture.read()

    cv2.imshow("Image", img)
    cv2.waitKey(1)
