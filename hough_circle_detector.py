# based on:
#  Robust Traffic Light and Arrow Detection Using Digital Map
#  Digital Map Based Signal State Recognition of Far Traffic Lights with Low Brightness

import cv2
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed

PLOT = False


# ----- Hough Circle parameters -----
dp = 1.2
minDist = 20
param1 = 50
param2 = 10     # >10 misses small further TL
minRadius = 0
maxRadius = 10  # >10 has a lot of big circles because of red/white lines


def hough_circle_detection(img, visualize):
    img_8bit = img.astype(np.uint8)

    # dp: inverse ratio of the accumulator resolution to the image resolution.
    #   the larger the dp gets, the smaller the accumulator array gets.
    # minDist: Minimum distance between the center (x, y) coordinates of detected circles.
    #   If the minDist is too small, multiple circles in the same neighborhood as the original may be (falsely) detected
    #   If the minDist is too large, then some circles may not be detected at all.
    # param1: Gradient value used to handle edge detection in the Yuen et al. method.
    # param2: Accumulator threshold value for the cv2.HOUGH_GRADIENT method.
    #   The smaller the threshold is, the more circles will be detected (including false circles).
    #   The larger the threshold is, the more circles will potentially be returned.
    # minRadius: Minimum size of the radius (in pixels).
    # maxRadius: Maximum size of the radius (in pixels).

    circles = cv2.HoughCircles(image=img_8bit, method=cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist,
                               param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # ensure at least some circles were found
    if circles is None:
        return None

    print(f"found {circles.shape[1]} circles")

    if visualize:
        output = img.copy()
        circles = np.uint16(np.around(circles[0, :])).astype(np.uint16)  # 1st dim of circles is useless
        for (x, y, r) in circles:
            # draw the outer circle
            cv2.circle(output, (x, y), r, (255, 0, 0), 2)
            # draw the center of the circle
            # cv.circle(cimg, (x, y), 2, (0, 0, 255), 3)
            # print(f"Circle at ({x}, {y}), r = {r}")

        cv2.imshow('detected circles', output)
        cv2.waitKey(0)

    return circles


# ----- Circle HSV filtering-----
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])
lower_green = np.array([40, 50, 50])
upper_green = np.array([90, 255, 255])
lower_yellow = np.array([15, 150, 150])
upper_yellow = np.array([35, 255, 255])
min_x_size_fraction = 0.1
max_x_size_fraction = 0.9
min_y_size_fraction = 0.1
max_y_size_fraction = 0.9
color_pixels_ratio = 0.5


def filter_circles(circles, image, color, visualize):
    if circles is None:
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if color == "red":
        mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    elif color == "green":
        mask = cv2.inRange(hsv, lower_green, upper_green)
    elif color == "yellow":
        mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    else:
        print(f"Unknown color {color}")

    min_x = min_x_size_fraction * image.shape[1]
    max_x = max_x_size_fraction * image.shape[1]
    min_y = min_y_size_fraction * image.shape[0]
    max_y = max_y_size_fraction * image.shape[0]

    filtered_circles = list()
    for circle in circles:
        if (circle[0] < min_x or circle[0] > max_x or circle[1] < min_y or circle[1] > max_y):
            continue

        r = int(circle[2])
        total_pixels = 0
        color_pixels = 0

        for x in (circle[0] + np.arange(-r, r + 1)):
            for y in (circle[1] + np.arange(-r, r + 1)):
                if x < image.shape[1] and x >= 0 and y < image.shape[0] and y > 0:
                    color_pixels += mask[y, x]
                    total_pixels += 1

        if color_pixels / total_pixels > 0.5:
            filtered_circles.append(circle)

    print(f"After filtering,  {len(filtered_circles)} circles")

    return filtered_circles


def visualize_circles(image, red_circles, green_circles, yellow_circles):
    cimg = image

    for circle in red_circles:
        cv2.circle(cimg, (circle[0], circle[1]), circle[2]+5, (0, 0, 255), 2)
    for circle in green_circles:
        cv2.circle(cimg, (circle[0], circle[1]), circle[2]+5, (0, 255, 0), 2)
    for circle in yellow_circles:
        cv2.circle(cimg, (circle[0], circle[1]), circle[2]+5, (0, 255, 255), 2)

    cv2.imshow('circles', cimg)
    cv2.waitKey(0)
