#!/usr/bin/env python3
import cv2

from hough_circle_detector import hough_circle_detection, filter_circles, visualize_circles
from hsv_highlighting import hsv_highlighting
from hsv_thresholding import hsv_threshold

VISUALIZE = True


def main():
    image = cv2.imread('../imgs/1280x720/6.png')
    # image = cv2.imread('imgs/test2.png')

    print(image.shape)
    cv2.imshow('image', image)
    # cv2.waitKey(0)

    # hsv_threshold(image, VISUALIZE)

    green, red, yellow = hsv_highlighting(img=image, visualize=False)

    # cv2.imshow('red', red)
    cv2.waitKey(0)
    circles_red = hough_circle_detection(red, VISUALIZE)
    circles_green = hough_circle_detection(green, VISUALIZE)
    circles_yellow = hough_circle_detection(yellow, VISUALIZE)

    circles_red = filter_circles(circles_red, image, "red", VISUALIZE)
    circles_green = filter_circles(circles_red, image, "green", VISUALIZE)
    circles_yellow = filter_circles(circles_red, image, "yellow", VISUALIZE)

    visualize_circles(image, circles_red, circles_green, circles_yellow)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
