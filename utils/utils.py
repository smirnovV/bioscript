import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def circle(r):
    a = np.arange(-r, r + 1) ** 2
    dists = np.sqrt(a[:, None] + a)
    return (np.abs(dists - r) < 0.5).astype(int)


def show(title, img):
    plt.imshow(img)
    plt.title(title)
    plt.show()


def to_RGB(img):
    return cv.cvtColor(img, cv.COLOR_GRAY2RGB)


def to_gray(img):
    return cv.cvtColor(img, cv.COLOR_RGB2GRAY)


def hist(img):
    return cv.calcHist([img], [0], None, [256], [0, 256])


def show_hist(ar):
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.plot(ar)
    plt.xlim([0, 256])
    plt.show()
