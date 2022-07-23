import math
import numpy as np


def polar(inside, outside, iris):
    height, width = 128, 256
    image = np.zeros((height, width), np.uint8)

    xin, yin, rin = inside['x'], inside['y'], inside['r']
    xon, yon, ron = outside['x'], outside['y'], outside['r']
    c1, c2 = 1.0 * (1 / height), 2.0 * (math.pi / width)

    for y in range(height):
        for x in range(width):
            p, o = c1 * y, c2 * x
            old_x = int(round((1 - p) * (xon + ron * math.cos(o)) + p * (xin + rin * math.cos(o))))
            old_y = int(round((1 - p) * (yon + ron * math.sin(o)) + p * (yin + rin * math.sin(o))))

            image[y, x] = iris[old_x, old_y]

    return image


def cut(outside, iris):
    x, y, r = outside['x'], outside['y'], outside['r']

    return iris[x - r: x + r, y - r: y + r]