import numpy as np
import cv2
from scipy.misc import imread

"""
Box => [smallX, bigX, smallY, bigY]
Bound => [
        [smallX, smallY], [smallX, bigY],
        [bigX, bigY], [bigX, smallY]
    ] or any other straight path order
Rectangle => [smallX, smallY, width, height]
"""


def _contour_to_rect(cnt):
    return cv2.boundingRect(cnt)


def contours_to_rects(contours):
    return np.unique(np.asarray([_contour_to_rect(cnt) for cnt in contours]), axis=0)


def _contour_to_bound(cnt):
    return rect_to_bound(_contour_to_rect(cnt))


def contours_to_bounds(contours):
    return np.unique(np.asarray([_contour_to_bound(cnt) for cnt in contours]), axis=0)


def _contour_to_box(cnt):
    rect_to_box(_contour_to_rect(cnt))


def contours_to_boxes(contours):
    return np.unique(np.asarray([_contour_to_box(cnt) for cnt in contours]), axis=0)


def rect_to_box(rectangle):
    return [rectangle[0], rectangle[0] + rectangle[2], rectangle[1], rectangle[1] + rectangle[3]]


def rect_to_bound(rectangle):
    return box_to_bound(rect_to_box(rectangle))


def box_to_rect(box):
    return [box[0], box[2], box[1] - box[0], box[3] - box[2]]


def bound_to_rect(bound):
    return box_to_rect(bound_to_box(bound))


def bound_to_box(bound):
    return [sorted([i[0] for i in bound])[i] for i in [0, 2]] + [sorted([i[1] for i in bound])[i] for i in [0, 2]]


def box_to_bound(box):
    return np.int0([[box[0], box[2]], [box[0], box[3]],
                    [box[1], box[3]], [box[1], box[2]]])


def is_intersect(box1, box2):
    return (box1[0] < box2[0] < box1[1] or box2[0] < box1[0] < box2[1]) and \
           (box1[2] < box2[2] < box1[3] or box2[2] < box1[2] < box2[3])


def box_intersection(box1, box2):
    return sorted(box1[:2] + box2[:2])[1:3] + sorted(box1[2:] + box2[2:])[1:3] if is_intersect(box1, box2) \
        else [0, 0, 0, 0]


def sort_box(box):
    return sorted(box[:2]) + sorted((box[2:]))


def box_area(box):
    return abs((box[0] - box[1]) * (box[2] - box[3]))


def box_union(box1, box2):
    return box_area(box1) + box_area(box2) - box_intersection(box1, box2)


def jaccard(rect1, rect2):
    inters = box_area(box_intersection(rect1, rect2))
    return inters / (box_area(rect1) + box_area(rect2) - inters)


def read_image(img, mode='RGB'):
    return imread(img, mode=mode) if isinstance(img, str) else img if type(img) == np.ndarray else None


def is_gray(image):
    return len(image.shape) == 2 and np.any(image > 1)


def gray(image):
    gray_image = imread(image, mode='L') if isinstance(image, str) else image if is_gray(image) else \
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def threshold(image, thresh_val=127, max_val=255, style=cv2.THRESH_BINARY):
    _, thresh_image = cv2.threshold(gray(image), thresh_val, max_val, style)
    return thresh_image


def hsv(image):
    hsv_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2HSV) if isinstance(image, str) else \
        cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image


def blur(image, blur_filter=5):
    image = cv2.imread(image) if isinstance(image, str) else image
    blur_image = cv2.GaussianBlur(image, (blur_filter, blur_filter), 0)
    return blur_image


def resize(img, max_width=300, max_height=300):
    img = read_image(img)
    height, width = img.shape[:2]
    # only shrink if img is bigger than required
    if max_height < height or max_width < width:
        # get scaling factor
        scaling_factor = max_height / float(height)
        if max_width / float(width) < scaling_factor:
            scaling_factor = max_width / float(width)
        img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return img


def get_contours(image):
    _, contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contours_bounds(img, contours=None):
    img = cv2.imread(img) if isinstance(img, str) else img
    img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)
    return img
