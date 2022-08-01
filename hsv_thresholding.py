import cv2
import numpy as np

from IPython import embed


def hsv_threshold(img, visualize):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # good
    S_thresh = (165, 230)
    V_thresh = (240, 255)
    # relaxed
    # S_thresh = (165, 200)
    # V_thresh = (30, 50)

    if visualize:
        # cv2.imshow('H', hsv[:, : , 0])
        cv2.imshow('S', hsv[:, :, 1])
        cv2.imshow('V', hsv[:, :, 2])
        cv2.waitKey(0)

    # ----- method 1 - cv2.inRange() -----
    # returns a binary result, 255 for pixels falling between the 2 thresholds, 0 otherwise
    S_threshold = cv2.inRange(hsv[:, :, 1], S_thresh[0], S_thresh[1])
    V_threshold = cv2.inRange(hsv[:, :, 2], V_thresh[0], V_thresh[1])

    # lower_blue = np.array([100, 50, 50])
    # upper_blue = np.array([150, 255, 255])
    # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # res = cv2.bitwise_and(img, img, mask=mask)

    if visualize:
        cv2.imshow('S_threshold', S_threshold)
        cv2.imshow('V_threshold', V_threshold)
        cv2.waitKey(0)

    # we need to be careful to not overflow, as by default images use uint8. if over/underflow values will be cut off
    # option 1: use float32
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    # ----- method 2 - udacity method (manual create mask) -----
    # s_binary = np.zeros_like(hsv[:, : , 1])
    # s_binary[(hsv[:, : , 1] >= S_thresh[0]) & (hsv[:, : , 1] <= S_thresh[1])] = 255
    # print(s_binary.nonzero())
    # cv2.imshow('s_binary', s_binary)
    # cv2.waitKey(0)

    # v_binary = np.zeros_like(hsv[:, : , 2])
    # v_binary[(hsv[:, : , 2] >= V_thresh[0]) & (hsv[:, : , 2] <= V_thresh[1])] = 255
    # cv2.imshow('v_binary', v_binary)
    # cv2.waitKey(0)

    # SV_binary = np.multiply(s_binary, v_binary) / 255 # element wise multiplication
    # cv2.imshow('SV_binary', SV_binary)
    # cv2.waitKey(0)

    return S_threshold, V_thresh

