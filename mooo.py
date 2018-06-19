


'''
import os
import cv2
import numpy as np
from package.plate_detection.detect_plate import PlateDetection
from scipy import ndimage
import operator
import skew_detect
import deskew


def horizontal_project(image, rows):
    max_rows = 0
    maximum = 0
    total_sum = 0.0
    horizontal_proj = np.zeros(rows, np.uint8)

    for (r, c), element in np.ndenumerate(image):
        if c == 0:
            continue
        difference = abs(image[r][c] - image[r][c - 1])
        if difference > 0:
            horizontal_proj[r] = horizontal_proj[r] + 1
            total_sum = total_sum + 1

            # peak
        if horizontal_proj[r] > maximum:
            max_rows = r
            maximum = horizontal_proj[r]

    return horizontal_proj, maximum, max_rows, total_sum


def vertical_projection(image, cols):
    max_vertical = 0
    maximum = 0
    total_sum = 0.0
    vertical_proj = np.zeros(cols, np.uint8)

    for (r, c), element in np.ndenumerate(image):
        if r == 0:
            continue
        difference = abs(image[r][c] - image[r - 1][c])
        if difference > 0:
            vertical_proj[c] = vertical_proj[c] + 1
            total_sum = total_sum + 1

            # peak
        if vertical_proj[c] > maximum:
            max_vertical = c
            maximum = vertical_proj[c]
    return vertical_proj, maximum, max_vertical, total_sum


def projection_filter(array, image, thresh_hold, cuts, iter_dims, horizontal):
    cut = False
    row_count = 0
    bound = np.zeros((cuts,), dtype=[('x', 'i4'), ('y', 'i4')])
    max_row = np.zeros(cuts, np.uint8)
    min_index = 0
    min_value = -1

    for fixed_dim, element in enumerate(array):
        if element <= thresh_hold:
            array[fixed_dim] = 0

            for iter_dim in range(iter_dims):
                if horizontal:
                    image[fixed_dim][iter_dim] = 0
                else:
                    image[iter_dim][fixed_dim] = 0

            if cut:
                lower_bound = fixed_dim
                cut = False
                if row_count >= min_value:
                    bound[min_index] = (lower_bound - row_count, lower_bound)
                    min_index, min_value = min(enumerate(max_row), key=operator.itemgetter(1))
                    row_count = 0
        else:
            row_count = row_count + 1
            if row_count >= min_value:
                max_row[min_index] = row_count
        cut = True
    return array, image, bound, max_row


def resize_box_large(img_thresh, box):
    (left, right) = (box[0], box[1])

    before = -1
    freq_same = 0
    thresh = 5
    freq_thresh = 2
    shift = int((right - left) * 5 / 100)
    length = image.shape[1]

    for i in range(shift):
        if left + i < length - 1:
            freq = 0

            for pixel in img_thresh[:, left + i]:

                if not (pixel == before):
                    if freq_same >= thresh:
                        freq += 1
                    freq_same = 0
                else:
                    freq_same += 1

                before = pixel

            if freq <= freq_thresh:
                box[0] = left + shift

                if box[0] >= length:
                    box[0] = length - 1
                break

    before = -1
    freq_same = 0

    for i in range(shift):
        if right - i > 0:
            freq = 0
            for pixel in img_thresh[:, right - i]:
                if not (pixel == before):
                    if freq_same >= thresh:
                        freq += 1
                    freq_same = 0
                else:
                    freq_same += 1
                before = pixel
            if freq <= freq_thresh:
                box[1] = right - shift
                if box[1] <= 0:
                    box[1] = 0
                break


def resize_box_small(img_thresh, box):
    (left, right) = (box[0], box[1])

    before = -1
    freq = 0
    freq_same = 0
    thresh = 5
    freq_thresh = 2
    shift = int((right - left) * 5 / 100)

    while True:

        if freq <= freq_thresh:
            break

        for i in range(shift):
            if left - i > 0:
                freq = 0
                for pixel in img_thresh[:, left - i]:

                    if not (pixel == before):
                        if freq_same >= thresh:
                            freq += 1
                        freq_same = 0
                    else:
                        freq_same += 1
                    before = pixel

                if freq > freq_thresh:
                    box[0] = left - shift
                    if box[0] < 0:
                        box[0] = 0
                    break

    before = -1
    freq = 0
    freq_same = 0
    length = image.shape[1]

    while True:

        if freq <= freq_thresh:
            break

        for i in range(shift):
            if right + i < length:
                freq = 0
                for pixel in img_thresh[:, right + i]:

                    if not (pixel == before):
                        if freq_same >= thresh:
                            freq += 1
                        freq_same = 0
                    else:
                        freq_same += 1
                    before = pixel

                if freq > freq_thresh:
                    box[1] = right + shift
                    if box[1] >= length:
                        box[1] = length - 1
                    break


def fix_skew(img_gray):
    try:
        res = skew_detect.determine_skew(img_gray)
        angle = res['Estimated Angle']
    except:
        angle = -90

    skew_img = deskew.deskew(img_gray, angle, 0)
    if skew_img.shape[0] > skew_img.shape[1]:
        skew_img = deskew.deskew(img_gray, 180, 0)

    skew_img = np.array(skew_img * 255, dtype=np.uint8)

    return skew_img, angle


def scfilter(image, iterations, kernel):
    for n in range(iterations):
        image = ndimage.filters.uniform_filter(image, size=kernel)
    return image


import pandas

images_names = []
var = pandas.read_csv(
    'C:/Users/324/Downloads/Compressed/models/research/object_detection/yousefdt/after renumber/test_labels.csv')
length = len(var['filename'])
bbox = np.ones([length, 9, 4])
labels = []
labels_array = []

last = ""
j = -1
num = 0
bit = 0
for i in range(length):
    if last == var['filename'][i]:
        num += 1

        bbox[j][num][0] = var['xmin'][i]
        bbox[j][num][1] = var['xmax'][i]
        bbox[j][num][2] = var['ymin'][i]
        bbox[j][num][3] = var['ymax'][i]
        labels.append(var['class'][i])

    else:
        labels_array.append(labels)
        j += 1
        num = 0
        labels = []
        bbox[j][num][0] = var['xmin'][i]
        bbox[j][num][1] = var['xmax'][i]
        bbox[j][num][2] = var['ymin'][i]
        bbox[j][num][3] = var['ymax'][i]
        labels.append(var['class'][i])
        images_names.append(str(var['filename'][i]) + ".jpg")

    last = var['filename'][i]

pd = PlateDetection()

path_img = "C:/Users/324/Downloads/Compressed/models/research/object_detection/yousefdt/after renumber/test/"
path_save = "C:/Users/324/Downloads/Compressed/models/research/object_detection/yousefdt/after renumber/check2/"
from imgaug import augmenters as iaa
import imgaug as ia

images_be = 0
images_af = 0
for f in range(length):
    name, ext = os.path.splitext(images_names[f])
    if ext == '.jpg':
        [box, img], prob, conf = pd.find(path_img + images_names[f])

        if box is None:
            continue

        # ymax = bbox[f][0][3]
        # ymin = bbox[f][0][2]
        # xmax = bbox[f][0][1]
        # xmin = bbox[f][0][0]

        # bb1 = ia.BoundingBox(x1=int(xmin), x2=int(xmax), y1=int(ymin), y2=int(ymax))
        # bb2 = ia.BoundingBox(x1=box[0], x2=box[1], y1=box[2], y2=box[3])
        # image_bbs = bb1.draw_on_image(image_bbs, thickness=2, color=[0, 255, 0])
        # cv2.imshow("1",image_bbs)
        # cv2.waitKey()
        # image_bbs = bb2.draw_on_image(image_bbs, thickness=2, color=[0, 255, 0])
        # cv2.imshow("2", image_bbs)

        # iou = bb1.iou(bb2)
        # print(iou)
        # if iou >= 0.5:
        #    images_be += 1
        # continue

        image_orig = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # cv2.imwrite(path_save+images_names[f], image_orig)

        # img_gray, angle = fix_skew(img_gray)

        img_gray = cv2.bitwise_not(image_orig)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

        image = cv2.threshold(img_gray[box[2]:box[3], box[0]:box[1]], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        image = np.copy(image_close)
        cols = image.shape[1]
        rows = image.shape[0]

        # apply horizontal projection
        horizontal_projection, _, max_col, _ = horizontal_project(image, rows)

        # Filter on the projection result
        horizontal_projection = scfilter(horizontal_projection, 1, 2)

        # max-min normalization on projection
        minimum = min(horizontal_projection)
        maximum = max(horizontal_projection)

        horizontal_projection = (horizontal_projection - minimum) * 1.0 / (maximum - minimum) * 1.0

        total_sum = 0
        for e in horizontal_projection:
            total_sum = total_sum + e
        average = total_sum / rows

        secondMin = np.nanmin(horizontal_projection)
        # get the biggest area with high freq

        horizontal_projection, _, horizontal_points, freq_array = projection_filter(horizontal_projection,
                                                                                    image,
                                                                                    average - secondMin, 2,
                                                                                    cols, True)
        # get the index of maximum freq
        max_index, _ = max(enumerate(freq_array), key=operator.itemgetter(1))
        row_upper = horizontal_points[max_index][0] - 10
        row_lower = horizontal_points[max_index][1]
        if row_upper < 0:
            row_upper = 0

        if row_lower >= rows:
            row_lower = rows - 1


        cv2.imwrite(path_save + images_names[f], image_orig[box[2]:box[3], box[0]:box[1]])
        print(box)
        box[2] = box[2] + row_upper
        box[3] = box[3] - (rows - row_lower-1)

        if box[2] == box[3]:
            box[2] = box[2] - row_upper
            box[3] = box[3] + (rows - row_lower-1)

        cv2.imwrite(path_save + name+ "h" + images_names[f], image_orig[box[2]:box[3], box[0]:box[1]])

        image = cv2.threshold(img_gray[box[2]:box[3], :], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        image_close = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        resize_box_large(image_close, box)
        resize_box_small(image_close, box)

        horizontal_strip = np.copy(image_orig[box[2]:box[3], box[0]:box[1]])
        cv2.imwrite(path_save + name + "l" + images_names[f], horizontal_strip)

# print((images_be/length) *100)
'''
