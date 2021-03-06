import numpy as np
import cv2


def binary_otsu(img_gray):
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_gray = cv2.bitwise_not(img_gray)
    img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    pixels_total = img_thresh.shape[1] * img_thresh.shape[0]

    nonzero = np.count_nonzero(img_thresh)
    if nonzero - (9/pixels_total) * 100 > pixels_total - nonzero:
        img_thresh = cv2.bitwise_not(img_thresh)
        # img_thresh = cv2.bitwise_not(img_thresh)

    '''
    gaussian = cv2.getGaussianKernel(3, 3)
    laplacian = np.array((
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]), dtype="int")
    sharpen = np.array(([-0.1, 0, -0.1],
                        [0.1, 2, 0.1],
                        [-0.1, -1, -0.1]), dtype="int")
    img_thresh= cv2.filter2D(img_thresh,-1,kernel=sharpen)
    #img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    #img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1)))
    img_thresh = cv2.GaussianBlur(img_thresh, (3, 3), 0)
    '''

    return img_thresh


def binarize(image, boxes, classes, scores):
    images = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im_width = image.shape[1]
    im_height = image.shape[0]
    boxes_under_thresh = 3
    index = 0
    temp_boxes = np.copy(boxes)
    for box in temp_boxes:
        boxes_under_current = 0
        (left, right, top, bottom) = (box[1] * im_width, box[3] * im_width, box[0] * im_height, box[2] * im_height)
        for box_under in temp_boxes:
            (left_under, right_under, top_under, bottom_under) = (box_under[1] * im_width,
                                                                  box_under[3] * im_width,
                                                                  box_under[0] * im_height,
                                                                  box_under[2] * im_height)
            if bottom <= top_under:
                boxes_under_current += 1

        if boxes_under_current > boxes_under_thresh:
            del classes[index]
            del scores[index]
            del boxes[index]
            continue

        temp = image[int(round(top - 2)):int(round(bottom + 2)), int(round(left)):int(round(right))]
        binary = binary_otsu(temp)
        images.append(binary)
        index += 1

    reverse = False
    for index_box in range(len(boxes)):
        if reverse:
            classes[index_box] = 1
        elif int(classes[index_box]) == 1:
            reverse = True

    return images
