import cv2
import math
import os
from imutils import paths
import argparse
import sys


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()


def video_framing(path):
    list_of_frames = []
    cap = cv2.VideoCapture(path)
    frame_rate = cap.get(5)  # frame rate

    while cap.isOpened():

        frame_id = cap.get(1)  # current frame number
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % math.ceil(frame_rate) == 0:
            list_of_frames.append(frame)

    cap.release()
    return detect_blur(list_of_frames)


def detect_blur(list_of_frames):
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    filtered_list_of_frames = []
    i = 1
    for image in list_of_frames:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        # text = "Not Blurry"
        # if the focus measure is less than the supplied threshold,
        # then the image should be considered "blurry"
        if not fm < 100:  # default threshold equals 100 (adjustable)
            filtered_list_of_frames.append(image)
    # filename = imagesFolder + "/image_" +  str(i) + ".jpg"
    # cv2.imwrite(filename, frame)
    # i=i+1
    return filtered_list_of_frames


if __name__ == '__main__':
    print(len(video_framing()))
