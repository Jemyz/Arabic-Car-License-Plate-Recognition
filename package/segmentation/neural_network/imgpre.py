import cv2
import numpy as np
from package.segmentation.neural_network import config


def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 0, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


def binary(img_gray):
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_gray = cv2.bitwise_not(img_gray)
    img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img_thresh


def remove_strip(img_gray, horizontal_strip):
    cols = 140
    start = 440
    cols_strip = horizontal_strip.shape[1]
    projection_white = np.zeros(cols_strip, np.uint8)
    projection_binary = np.zeros(cols, np.uint8)
    last_start = 0
    sum_col = 0
    start_flag = False

    for (r, c), element in np.ndenumerate(img_gray):
        if img_gray[r][c] == 255:
            projection_white[c] += 1

        if horizontal_strip[r][c] == 255 and c >= start and c < start + cols:
            projection_binary[c - start] = 1

    for c, element in enumerate(projection_binary):
        if projection_binary[c] == 0:
            if sum_col > 0 and sum_col < config.strip_thresh and start_flag:
                projection_binary[c] = 254
                projection_binary[last_start] = 255
            sum_col = 0
            start_flag = True
            last_start = c
        else:
            sum_col += 1
            projection_binary[c] = 1

    start_flag = False

    for (r, c), element in np.ndenumerate(img_gray):

        if projection_white[c] < config.white_thresh and (
                (c >= start - 20 and c < start + cols) or c < 75 or c > config.frame_end_thresh):
            horizontal_strip[r][c] = 0
        if c >= start and c < start + cols and projection_binary[c - start] == 255:
            start_flag = True
        if c >= start and c < start + cols and projection_binary[c - start] == 254:
            start_flag = False
        if start_flag:
            horizontal_strip[r][c] = 0
    return horizontal_strip
