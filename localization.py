from utils.utils import to_gray, circle
import cv2 as cv
import numpy as np


def inside_border(img):
    min_r, max_r, thresh_value, rect_size, sobel_size, distance = 25, 75, 55, 7, 3, 75

    img = to_gray(img)
    ret, thresh = cv.threshold(img, thresh_value, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(rect_size, rect_size))
    ero = cv.erode(thresh, kernel, iterations = 1)
    dil = cv.dilate(ero, kernel, iterations = 1)
    sob = cv.Sobel(dil, 0, 1, 1, ksize = sobel_size, scale = 1, delta = 1, borderType = cv.BORDER_DEFAULT)

    a, b = img.shape
    l = list(np.zeros((a,b)) for r in range(max_r - min_r))
    for x in range(distance, a - distance):
        for y in range(distance, b - distance):
            if sob[x, y] == 255:
                for r in range(min_r, max_r):
                    if 0 < x - r and r + x < a and 0 < y - r and r + y < b:
                        mat = np.zeros((a, b))
                        mat[x - r: x + r + 1, y - r  : y + r + 1] = circle(r)
                        l[r - min_r] += mat
    x, y , r, s = 0, 0, 0, 0

    for ind, value in enumerate(l):
        x1, y1 = np.unravel_index(value.argmax(), value.shape)
        if s < value[x1, y1]:
            x, y, s, r = x1, y1, value[x1, y1], ind + min_r

    return {'x' : x, 'y' : y , 'r' : r}


def outside_border(inside, img):
    img = to_gray(img)
    min_r, max_r, thresh_value, rect_size, sobel_size, distance = inside['r'] * 2, inside['r'] * 4, 140, 7, 3, 75

    ret, thresh = cv.threshold(img, thresh_value, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(21, 21))
    ero = cv.erode(thresh, kernel, iterations = 1)
    dil = cv.dilate(ero, kernel, iterations = 1)
    sob = cv.Sobel(dil, 0, 1, 1, ksize = sobel_size, scale = 1, delta = 1, borderType = cv.BORDER_DEFAULT)
    a, b = img.shape
    l = list(np.zeros((a,b)) for r in range(max_r - min_r))
    for x in range(0, a):
        for y in range(0, b):
            if sob[x, y] == 255:
                for r in range(min_r, max_r):
                    mat = np.zeros((a, b))
                    x_1, x_2, y_1, y_2 = max(x - r, 0), min(x + r + 1, a), max(y - r, 0), min(y + r + 1, b)
                    mat[x_1 : x_2, y_1 : y_2] = circle(r)[abs(min(x - r, 0)): abs(min(x - r, 0)) + x_2 - x_1, abs(min(y - r, 0)): abs(min(y - r, 0)) + y_2 - y_1]
                    l[r - min_r] += mat
    x, y , r, s = 0, 0, 0, 0

    for ind, value in enumerate(l):
        x1, y1 = np.unravel_index(value.argmax(), value.shape)
        if s < value[x1, y1]:
            x, y, s, r = x1, y1, value[x1, y1], ind + min_r

    return {'x' : x, 'y' : y , 'r' : r}