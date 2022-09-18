import cv2
import numpy as np
import HandTrackingModule as htm
import time

# Library that allows us to move around with mouse
import autopy

widthCamera = 640
heightCamera = 480
frameReduction =100
smoothening = 7

pTime = 0
prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0

# Taking input from camera and setting height and width for the frame
capture = cv2.VideoCapture(0)  # 0: External WebCam and 1 Other
capture.set(3, widthCamera)
capture.set(4, heightCamera)

# Detect hand marks
detector = htm.handDetector(maxHands=1)  # Expecting only 1 hand

# To get width and height of screen
widthScreen, heightScreen = autopy.screen.size()
# print(widthScreen, heightScreen)



while True:
    # 1. Find hand Landmarks
    success, img = capture.read()
    img = detector.findHands(img)

    # Detect positions of the hands and to draw it
    lmList, boundingBox = detector.findPosition(img)

    # 2. Get the tip of the index and middle finger
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]  # Points for index finger
        x2, y2 = lmList[12][1:]  # Points for Middle finger

        # print(x1, y1, x2, y2)  #  For testing

    # 3. Check which fingers are up
    if len(lmList) != 0:
        fingers = detector.fingersUp()
        # print(fingers)

        cv2.rectangle(img, (frameReduction, frameReduction),
                      (widthCamera - frameReduction, heightCamera - frameReduction), (255, 0, 255), 2)
        # 4. Only Index finger up means it is in moving mode

        # check if only index fingers are up
        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameReduction, widthCamera - frameReduction),
                           (0, widthScreen))  # Converting coordinates
            y3 = np.interp(x1, (frameReduction, heightCamera - frameReduction),
                           (0, heightScreen))  # Converting coordinates

            # 5. Smoothen values to remove jitter and flicker
            currLocX = prevLocX + (x3 - prevLocX) / smoothening
            currLocY = prevLocY + (y3 - prevLocY) / smoothening

            # 6. Move mouse using autopy
            autopy.mouse.move(widthScreen - currLocX, currLocY)  # widthScreen-x3 to flip
            cv2.circle(img, (x1, y1), 14, (255, 0, 255),
                       cv2.FILLED)  # Pink circle in index fingertip during moving mode
            prevLocX, prevLocY = currLocX, currLocY

        # 7. Check if we are in clicking mode: If both middle and index fingers are up: Click Mode
        if fingers[1] == 1 and fingers[2] == 1:
            length, img, lineInformation = detector.findDistance(8, 12, img)  # Landmark 8 & 12
            # print(length)

            #  Click if length less than 40
            if length < 40:
                cv2.circle(img, (lineInformation[4], lineInformation[5]), 14,
                           (0, 255, 0), cv2.FILLED)  # Green color if clicked

                # Do click
                autopy.mouse.click()
    else:
        fingers = [0, 0, 0, 0, 0]

    # 11. Frame Rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)