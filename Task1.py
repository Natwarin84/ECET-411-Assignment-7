# Clone the red and blue detection repository from GitHub
# !git clone https://github.com/Majdawad88/red_and_blue_detection.git

import cv2 # Import OpenCV library for image processing
import numpy as np # Import NumPy for numerical and array operations
import time # Import time for handling delays and sleep
import sys # Import sys for system-specific parameters and functions
from picamera2 import Picamera2 # Import the Picamera2 library to control the Raspberry Pi camera

# --- Configuration ---

# Define the lower boundary for the first range of Red in HSV (H: 0-10)
lower_red1 = np.array([0, 120, 70])
# Define the upper boundary for the first range of Red in HSV
upper_red1 = np.array([10, 255, 255])

# Define the lower boundary for the second range of Red (wraps around the spectrum H: 170-180)
lower_red2 = np.array([170, 120, 70])
# Define the upper boundary for the second range of Red in HSV
upper_red2 = np.array([180, 255, 255])

# Define the lower boundary for Blue in HSV
lower_blue = np.array([100, 150, 0])
# Define the upper boundary for Blue in HSV
upper_blue = np.array([140, 255, 255])

# Define Green range
lower_green = np.array([35, 80, 50])
upper_green = np.array([85, 255, 255])

# Define Yellow range
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([35, 255, 255])

# Define White range
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 40, 255])

# Define Black range
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 40])


# Set the dimensions for a single side (half) of the combined output window
HALF_W, HALF_H = 560, 420

# Calculate the total window width (double the half width) and total height
DISPLAY_W, DISPLAY_H = HALF_W * 2, HALF_H


# --- State ---

# Initialize the starting detection mode
mode = "RED"

# Global flag to track if the "QUIT" button has been clicked via mouse
quit_clicked = False

# Global flag to track if the "MODE" toggle button has been clicked via mouse
mode_clicked = False


# Function to handle mouse click events on the CV2 window
def mouse_callback(event, x, y, flags, param):

global quit_clicked, mode_clicked

if event == cv2.EVENT_LBUTTONDOWN:

# If click coordinates are within the "QUIT" button
if 10 <= x <= 110 and 10 <= y <= 50:
quit_clicked = True

# If click coordinates are within the "MODE" button
if 130 <= x <= 260 and 10 <= y <= 50:
mode_clicked = True


# --- Window Setup ---

win_name = "Color Tracker"
cv2.namedWindow(win_name)
cv2.setMouseCallback(win_name, mouse_callback)


# --- Camera Setup ---

picam2 = Picamera2()
picam2.configure("preview")
picam2.start()


try:
while True:

# Capture frame
frame_rgb = picam2.capture_array()

# Convert RGB to BGR for OpenCV
frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

# Flip the frame vertically
frame_bgr = cv2.flip(frame_bgr, 0)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(frame_bgr, (11, 11), 0)

# Convert image to HSV
hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


# --- Choose Mask ---

if mode == "RED":

m1 = cv2.inRange(hsv, lower_red1, upper_red1)
m2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = cv2.bitwise_or(m1, m2)
color_theme = (0, 0, 255)

elif mode == "BLUE":

mask = cv2.inRange(hsv, lower_blue, upper_blue)
color_theme = (255, 0, 0)

elif mode == "GREEN":

mask = cv2.inRange(hsv, lower_green, upper_green)
color_theme = (0, 255, 0)

elif mode == "YELLOW":

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
color_theme = (0, 255, 255)

elif mode == "WHITE":

mask = cv2.inRange(hsv, lower_white, upper_white)
color_theme = (255, 255, 255)

elif mode == "BLACK":

mask = cv2.inRange(hsv, lower_black, upper_black)
color_theme = (50, 50, 50)


# --- Contour Detection ---

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contours:

largest_contour = max(contours, key=cv2.contourArea)

if cv2.contourArea(largest_contour) > 500:

x, y, w, h = cv2.boundingRect(largest_contour)

cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 3)

cv2.putText(frame_bgr, f"Target {mode}", (x, y - 10),
cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# --- Draw UI Buttons ---

cv2.rectangle(frame_bgr, (10, 10), (110, 50), (0, 0, 200), -1)
cv2.putText(frame_bgr, "QUIT", (25, 40),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

cv2.rectangle(frame_bgr, (130, 10), (260, 50), (0, 200, 0), -1)
cv2.putText(frame_bgr, "MODE", (155, 40),
cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

cv2.putText(frame_bgr, f"CURRENT: {mode}", (10, 90),
cv2.FONT_HERSHEY_SIMPLEX, 1, color_theme, 2)


# --- Mode Switching Logic ---

if mode_clicked:

if mode == "RED":
mode = "BLUE"
elif mode == "BLUE":
mode = "GREEN"
elif mode == "GREEN":
mode = "YELLOW"
elif mode == "YELLOW":
mode = "WHITE"
elif mode == "WHITE":
mode = "BLACK"
else:
mode = "RED"

mode_clicked = False


# --- Display Combined View ---

left_img = cv2.resize(frame_bgr, (HALF_W, HALF_H))
right_img = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (HALF_W, HALF_H))

combined = np.hstack((left_img, right_img))

cv2.imshow(win_name, combined)


# Exit condition
if (cv2.waitKey(1) & 0xFF == ord('q')) or quit_clicked:
break


finally:

picam2.stop()
cv2.destroyAllWindows()
