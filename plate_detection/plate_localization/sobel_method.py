from plate_detection.misc import *
from plate_detection.misc import read_image
from cv2 import CV_8U


def __sobel_edge(image, horizontal=False, vertical=True, x_derivative=1, y_derivative=1, depth=CV_8U, k_size=3):
    """
    :param image: image need to find edges inside.
    :param horizontal: determine whether to include horizontal edges or not
    :param vertical: determine whether to include vertical edges or not
    :param x_derivative: The order of the derivative in x direction.
    :param y_derivative: The order of the derivative in y direction.
    :param depth: image depth
    :param k_size: must be odd number and smaller than 31
    :return: edge image
    """
    hor_edges = ver_edges = None
    assert horizontal or vertical, 'either horizontal or vertical should be selected.'
    if horizontal:
        hor_edges = cv2.Sobel(image, depth, 0, y_derivative, ksize=k_size)
    if vertical:
        ver_edges = cv2.Sobel(image, depth, x_derivative, 0, ksize=k_size)
    img = hor_edges + ver_edges if horizontal and vertical else hor_edges if horizontal else ver_edges
    return img


def sobel_edge_method(img, sizes, area_filter, area_ratio, area_range):
    """detect objects in image using edge sobel edge detection method"""
    img = read_image(img)
    gray_image = gray(img)
    ver_edges = __sobel_edge(gray_image)
    thresh = threshold(ver_edges)
    if sizes is None:
        sizes = [(min(16, int(img.shape[1] / 12)), min(4, int(img.shape[0] / 20)))]
    bounds = list()
    for size in sizes:
        se = cv2.getStructuringElement(cv2.MORPH_RECT, size)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)
        im2, contours, hierarchy = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounds += filter_bounds(img.shape, contours, area_ratio=area_ratio, area_range=area_range) if area_filter \
            else contours_to_bounds(contours).tolist()
    return np.unique(bounds, axis=0) if len(bounds) > 0 else bounds


def filter_bounds(img_shape, contours, area_ratio, area_range, bounds=None):
    """Filter bounds by area."""
    filtered_bounds = []
    area_ratio *= img_shape[0] * img_shape[1]
    if bounds is None:
        bounds = contours_to_bounds(contours)
    for bound in bounds:
        width = bound[2][0] - bound[0][0]
        height = bound[1][1] - bound[0][1]
        area = width * height
        if width > height and area_range[0] * area_ratio < area < area_range[1] * area_ratio:
            filtered_bounds.append(bound)
    return filtered_bounds
