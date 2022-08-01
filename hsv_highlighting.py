# based on:
#  Robust Traffic Light and Arrow Detection Using Digital Map
#  Digital Map Based Signal State Recognition of Far Traffic Lights with Low Brightness

import cv2
import numpy as np
import matplotlib.pyplot as plt

from IPython import embed

PLOT = False


def hsv_highlighting(img, visualize):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if visualize:
        # cv2.imshow('H', hsv[:, :, 0])
        cv2.imshow('S', hsv[:, :, 1])
        # cv2.imshow('V', hsv[:, :, 2])
        cv2.waitKey(0)

    # we need to be careful to not overflow, as by default images use uint8. if over/underflow values will be cut off
    # option 1: use float32
    H = hsv[:, :, 0].astype(np.float32)
    S = hsv[:, :, 1].astype(np.float32)
    V = hsv[:, :, 2].astype(np.float32)

    # ----- Normalization -----
    # -- normalize V --
    V_avg = np.average(V)
    V_std = np.std(V)

    kv = 1.2            # increase will highlight the brightness of the whole picture
    Vm_std = V_std/5.0  # reduce will make result smoother

    Vm = kv * V_avg + (Vm_std/V_std) * (V - V_avg)

    # kv>1 will cause overflow as Vm > 255
    # for low number (kv=2), this works ok when computing SVm, but hight numbers can mess it up

    # Vm2 = Vm * 255.0 / np.amax(Vm)    # this removes the overflow, but causes kv to have no effect

    # -- normalize S --
    # this is a weird sigmoid, after a certain value, the multiplecation by S dominates and it becomes almost a linear function (Sm = S)
    # results is almost a threshold function, but with a sigmoid at the threshold instead of a step threshold
    # result is much nicer than a step threshold function

    a = 0.5         # a increase => sigmoid slope is stronger
    b = 100         # value for which sigmoid = 0.5   =>   Sm(b) = 0.5 * S(b)

    Sm = S / (1 + np.exp(-a*(S - b)))

    if visualize:
        # cv2.imshow('Vm', Vm.astype(np.uint8))
        cv2.imshow('Sm', Sm.astype(np.uint8))
        # cv2.imshow('Vm2', Vm2.astype(np.uint8))
        cv2.waitKey(0)

    # -- Multiplication S × V /255 --

    SV = S * V
    SV = SV/255

    SVm = S * Vm / 255
    SmVm = Sm * Vm / 255
    # SVm2 = S * Vm2 / 255

    # option 2: pre-divide V by 255
    # V_norm = hsv[:, : , 2]/255
    # SV = hsv[:, : , 1] * V_norm
    # SV = SV.astype(np.uint8)    # re-case into uint8, otherwise result is wrong when showing image

    if visualize:
        # cv2.imshow('SV', SV.astype(np.uint8))
        # cv2.imshow('SVm', SVm.astype(np.uint8))
        cv2.imshow('SmVm', SmVm.astype(np.uint8))
        # cv2.imshow('SVm2', SVm2.astype(np.uint8))
        cv2.waitKey(0)

    w_g, w_r, w_y, w_t = H_weighting(H, visualize)

    SmVmWg = SmVm * w_g
    SmVmWr = SmVm * w_r
    SmVmWy = SmVm * w_y
    SmVmWt = SmVm * w_t

    if visualize:
        # cv2.imshow('SmVmWg', SmVmWg.astype(np.uint8))
        cv2.imshow('SmVmWr', SmVmWr.astype(np.uint8))
        # cv2.imshow('SmVmWy', SmVmWy.astype(np.uint8))
        # cv2.imshow('SmVmWt', SmVmWt.astype(np.uint8))
        cv2.waitKey(0)

    return SmVmWg.astype(np.uint8), SmVmWr.astype(np.uint8), SmVmWy.astype(np.uint8)


# ----- Hue weighting -----
# -- define the validity distributions --
# these values depends on camera's color balance
# in openCV H is in [0, 180], but the reference numbers are defined in [0, 360]
mu_g = 180 / 2
mu_y = 30 / 2
mu_r = 5 / 2
sigma_g = 20 / 2
sigma_y = 10 / 2
sigma_r = 20 / 2


# TODO these functions should be wrapped around [0, 180]
# however I couldn't manage to find a function that wrap them correctly for all
# for g, we just don't wrap as mu_g is near the center of [0,180] so the distribution should not overflow
def Wg(x):
    return np.exp(-(x-mu_g)**2/(2*sigma_g**2))


def Wr(x):
    x = ((x + 90) % 180) - 90
    return np.exp(-(x-mu_r)**2/(2*sigma_r**2))


def Wy(x):
    x = ((x + 90) % 180) - 90
    return np.exp(-(x-mu_y)**2/(2*sigma_y**2))


def H_weighting(H, visualize):
    w_g = Wg(H)
    w_r = Wr(H)
    w_y = Wy(H)
    w_t = np.maximum(w_g, np.maximum(w_r, w_y))

    if PLOT:
        x = np.arange(0., 180.0, 1.0)
        plt.plot(x, Wg(x), 'g')
        plt.plot(x, Wr(x), 'r')
        plt.plot(x, Wy(x), 'y')
        plt.grid()
        plt.show()

    if visualize:
        # w_xx values are [0, 1], so multiple by 255 for visualizing
        # cv2.imshow('Wg', (w_g*255).astype(np.uint8))
        cv2.imshow('Wr', (w_r*255).astype(np.uint8))
        # cv2.imshow('Wy', (w_y*255).astype(np.uint8))
        # cv2.imshow('Wt', (w_t*255).astype(np.uint8))
        cv2.waitKey(0)

    return w_g, w_r, w_y, w_t

