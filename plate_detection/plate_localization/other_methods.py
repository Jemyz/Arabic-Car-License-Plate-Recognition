from plate_detection.misc import *


def _canny_edge(image, min_len=100, max_len=150):
    """
    Find edges in an image using canny algorithm.
    :param image:
    :param min_len: any edge with length smaller than min_len will be discarded.
    :param max_len: any edge of length larger than max_len will be considered as valid edge.
    :return:
    """
    image = gray(cv2.imread(image)) if isinstance(image, str) else image
    # getting Binary Image using canny edge detection
    edged = cv2.Canny(image, min_len, max_len)
    return edged


def _morph_edge(image, horizontal=False, vertical=True):
    gray_image = gray(image)
    gray_image = cv2.bitwise_not(gray_image)
    bw = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    if horizontal:
        horizontal = np.copy(bw)
        cols = horizontal.shape[1]
        horizontal_size = int(cols / 10)
        # Create structure element for extracting horizontal lines through morphology operations
        horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
        # Apply morphology operations
        hor = cv2.erode(horizontal, horizontal_structure)
        hor = cv2.dilate(horizontal, hor)
    if vertical:
        vertical = np.copy(bw)
        rows = vertical.shape[0]
        vertical_size = int(rows / 20)
        # Create structure element for extracting vertical lines through morphology operations
        vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
        # Apply morphology operations
        vert = cv2.erode(vertical, vertical_structure)
        vert = cv2.dilate(vertical, vert)
    img = vert + hor if horizontal and vertical else vert if vertical else hor if horizontal else None
    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''
    # Step 1
    edges = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)
    # Step 2
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.dilate(edges, kernel)
    # Step 3
    smooth = np.copy(vertical)
    # Step 4
    smooth = cv2.blur(smooth, (2, 2))
    # Step 5
    (rows, cols) = np.where(edges != 0)
    img[rows, cols] = smooth[rows, cols]
    # Show final result
    cv2.imshow("smooth - final", img)
    return img


def _adapt_threshold(image, max_val=255, adapt_type=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, style=cv2.THRESH_BINARY,
                     blur_image=None):
    if blur_image is None:
        blur_image = blur(image)
    thresh_image = cv2.adaptiveThreshold(blur_image, max_val, adapt_type, style, 11, 2)
    return thresh_image


def _histo_equalize(image):
    image = cv2.imread(image) if isinstance(image, str) else image
    # Applying Histogram Equalization
    # visit https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
    histo_image = cv2.equalizeHist(image)
    return histo_image


def _normalize_image(image, show=False):
    image = cv2.imread(image) if isinstance(image, str) else image
    norm_image = cv2.normalize(image, dst=image, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if show:
        cv2.imshow("Normalized Image Image", norm_image)
    return norm_image


def prepare_image(image, show=False):
    """
    :param image:
    :param show:
    :return:
    """
    gray_image = gray(image)
    thresh_image = threshold(gray_image)
    cv2.imshow('asdasda', thresh_image)
    edges = _canny_edge(thresh_image)
    cv2.imshow('edges', edges)
    im2, contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv2.imshow('houghlines5.jpg', image)
    return _canny_edge(_histo_equalize(gray(image), show))


def hough_lines(image, show=False, edged=None, probalistic=False):
    if edged is None:
        edged = _canny_edge(image)
    if probalistic:
        lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 100, 100, 10)
        if show:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edged, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow('probabilistic hough lines', image)
    else:
        lines = cv2.HoughLines(edged, 1, np.pi / 180, 100)
        if show:
            for line in lines:
                rho, theta = line[0]
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * -b)
                y1 = int(y0 + 1000 * a)
                x2 = int(x0 - 1000 * -b)
                y2 = int(y0 - 1000 * a)
                cv2.line(edged, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.imshow('hough lines', image)
    return lines


def skeleton(img_src):
    img = read_image(img_src)
    gray_image = gray(img)
    thresh = threshold(gray_image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False
    skel = np.zeros(thresh.shape, dtype=np.uint8)
    size = np.size(img)
    while not done:
        eroded = cv2.erode(thresh, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(thresh, temp)
        skel = cv2.bitwise_or(skel, temp)
        thresh = eroded.copy()
        zeros = size - cv2.countNonZero(thresh)
        if zeros == size:
            done = True
    cv2.imshow("skel", skel)


def _draw_lines(img, lines):
    for rho, theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * -b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * -b)
        y2 = int(y0 - 1000 * a)
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return img


def nothing(x):
    pass


def color_method(img):
    cv2.namedWindow("Control", cv2.WINDOW_AUTOSIZE)
    lh = 100
    hh = 179
    ls = 0
    hs = 10
    lv = 255
    hv = 255
    cv2.createTrackbar("lh", "Control", lh, 179, nothing)
    cv2.createTrackbar("hh", "Control", hh, 179, nothing)
    cv2.createTrackbar("ls", "Control", ls, 255, nothing)
    cv2.createTrackbar("hs", "Control", hs, 255, nothing)
    cv2.createTrackbar("lv", "Control", lv, 255, nothing)
    cv2.createTrackbar("hv", "Control", hv, 255, nothing)
    img = hsv(img)
    img = resize(img)
    while True:
        img2 = img.copy()
        thresh = cv2.inRange(img, (lh, ls, lh), (hh, hs, hv))
        # thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # thresh = cv2.dilate(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # thresh = cv2.erode(thresh, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cv2.imshow('thresh', thresh)
        im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        draw_contours_bounds(img2, contours)
        cv2.imshow('final', img2)
        lh = cv2.getTrackbarPos('lh', 'Control')
        hh = cv2.getTrackbarPos('hh', 'Control')
        ls = cv2.getTrackbarPos('ls', 'Control')
        hs = cv2.getTrackbarPos('hs', 'Control')
        lv = cv2.getTrackbarPos('lv', 'Control')
        hv = cv2.getTrackbarPos('hv', 'Control')
        cv2.waitKey(0)