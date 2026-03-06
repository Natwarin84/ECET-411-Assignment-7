# ============================================================
# ECET 411 – Assignment 7
# Task 1: Multi-Mode Color Tracker
# Detects: Red, Blue, Green, Yellow, White, Black
# ============================================================

# ---------- IMPORT LIBRARIES ----------

import cv2                    # OpenCV for computer vision
import numpy as np            # NumPy for arrays
import time                   # time delay
from picamera2 import Picamera2   # Raspberry Pi camera


# ---------- COLOR RANGES (HSV) ----------

# Red (two ranges because red wraps around HSV spectrum)
lower_red1 = np.array([0,120,70])
upper_red1 = np.array([10,255,255])

lower_red2 = np.array([170,120,70])
upper_red2 = np.array([180,255,255])

# Blue
lower_blue = np.array([100,150,0])
upper_blue = np.array([140,255,255])

# Green
lower_green = np.array([40,70,70])
upper_green = np.array([80,255,255])

# Yellow
lower_yellow = np.array([20,100,100])
upper_yellow = np.array([30,255,255])

# White
lower_white = np.array([0,0,200])
upper_white = np.array([180,40,255])

# Black
lower_black = np.array([0,0,0])
upper_black = np.array([180,255,40])


# ---------- STARTING MODE ----------

mode = "RED"   # default detection mode


# ---------- CAMERA SETUP ----------

picam2 = Picamera2()         # create camera object
picam2.configure("preview")  # configure preview mode
picam2.start()               # start camera
time.sleep(0.3)              # short warm-up delay


# ---------- WINDOW ----------

cv2.namedWindow("Multi-Mode Color Tracker")


# ---------- MAIN LOOP ----------

while True:

    # capture frame from camera
    frame = picam2.capture_array()

    # convert RGB → BGR for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(frame,(11,11),0)

    # convert image to HSV color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


    # ---------- SELECT COLOR MODE ----------

    if mode == "RED":

        m1 = cv2.inRange(hsv, lower_red1, upper_red1)
        m2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(m1,m2)

    elif mode == "BLUE":

        mask = cv2.inRange(hsv, lower_blue, upper_blue)

    elif mode == "GREEN":

        mask = cv2.inRange(hsv, lower_green, upper_green)

    elif mode == "YELLOW":

        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    elif mode == "WHITE":

        mask = cv2.inRange(hsv, lower_white, upper_white)

    elif mode == "BLACK":

        mask = cv2.inRange(hsv, lower_black, upper_black)


    # ---------- FIND OBJECT CONTOURS ----------

    contours,_ = cv2.findContours(mask,
                                  cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)

    if contours:

        # select largest detected object
        largest = max(contours, key=cv2.contourArea)

        # ignore very small detections (noise)
        if cv2.contourArea(largest) > 500:

            x,y,w,h = cv2.boundingRect(largest)

            # draw rectangle around detected object
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

            # display active color mode
            cv2.putText(frame,f"Target: {mode}",
                        (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0,255,0),
                        2)


    # ---------- DISPLAY CAMERA ----------

    cv2.imshow("Multi-Mode Color Tracker", frame)


    # ---------- KEYBOARD CONTROLS ----------

    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'):
        mode = "RED"

    elif key == ord('b'):
        mode = "BLUE"

    elif key == ord('g'):
        mode = "GREEN"

    elif key == ord('y'):
        mode = "YELLOW"

    elif key == ord('w'):
        mode = "WHITE"

    elif key == ord('k'):
        mode = "BLACK"

    elif key == ord('q'):
        break


# ---------- CLEANUP ----------

picam2.stop()
cv2.destroyAllWindows()
