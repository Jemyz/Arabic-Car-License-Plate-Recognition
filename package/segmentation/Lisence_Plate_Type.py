import cv2
import numpy as np
import os
import glob


def get_lisence_type(img):  # Send cv2 image object

    # img = cv2.imread(f1)
    [h, w, c] = img.shape
    h = int((h / 3))
    w = w + 20
    red = 0
    green = 0
    blue = 0
    crop_img = img[8:h, 20:w - 20]
    [h, w, c] = crop_img.shape
    for y in range(0, h):
        for x in range(0, w):
            [b, g, r] = crop_img[y, x]
            blue = blue + b
            green = green + g
            red = red + r
    size = h * w
    # print(size)
    # Na2l-> red,blue,green . . . Ogra -> red,green,blue
    blue = blue / size
    green = green / size
    red = red / size
    if blue >= green and blue >= red:
        result = "ملاكى"
    elif red >= green and red >= blue:
        result = "نقل"
    else:
        result = "أجره"
    return result
