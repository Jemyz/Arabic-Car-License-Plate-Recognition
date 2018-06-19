import os
import cv2
import numpy as np
import tensorflow as tf
from package.plate_detection.misc import read_image

from package.plate_detection.localization_abstract import LocalizationAbstract


class ObjectDetection(LocalizationAbstract):

    def __init__(self, model="Inception"):
        # Path to image
        # PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)
        print("loading localization model")

        # Grab path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        path_to_model = os.path.join(self.CWD_PATH, "package", "plate_detection", "object_detection_plates", "models",
                                     model+'.pb')

        # Load the Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(graph=detection_graph)

        # Define input and output tensors (i.e. data) for the object detection classifier

        # Input tensor is the image
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensors are the detection boxes, scores, and classes
        # Each box represents a part of the image where a particular object was detected
        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        # Each score represents level of confidence for each of the objects.
        # The score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        # Number of objects detected
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    def resize_box(self, image, box):
        (left, right) = (box[0], box[1])
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        img_gray = cv2.bitwise_not(img_gray)
        img_thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img_thresh = img_thresh[box[2]:box[3]].copy()

        before = -1
        freq = 0
        freq_same = 0
        thresh = 5
        freq_thresh = 3
        shift = int((right - left) * 5 / 100)

        for i in range(shift):
            if left - i > 0:
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
                else:
                    freq = 0

        before = -1
        freq = 0
        freq_same = 0
        length = image.shape[1]

        for i in range(shift):
            if right + i < length:
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
                else:
                    freq = 0

    def find(self, image):
        image = read_image(image)

        # Perform the actual detection by running the model with the image as input
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        xs = np.argsort(boxes[0, :, 1])
        box_final = None
        class_final = None
        score_final = None

        im_width = image.shape[1]
        im_height = image.shape[0]
        maximum_height = 0
        last_index = -1

        for i in xs:

            if scores[0][i] > 0.8:
                (left, right, top, bottom) = (boxes[0][i][1] * im_width,
                                              boxes[0][i][3] * im_width,
                                              boxes[0][i][0] * im_height,
                                              boxes[0][i][2] * im_height)
                '''
                if classes[0][i] == 1:
                    cv2.rectangle(image,
                                  (int(round(left)), int(round(top))),
                                  (int(round(right)), int(round(bottom))),
                                  (255, 0, 0),
                                  4)

                if classes[0][i] == 2:
                    cv2.rectangle(image,
                                  (int(round(left)), int(round(top))),
                                  (int(round(right)), int(round(bottom))),
                                  (0, 0, 255),
                                  4)
                '''
                if (bottom - top) > maximum_height:

                    box_final = [int(round(left)), int(round(right)), int(round(top)), int(round(bottom))]
                    class_final = classes[0][i]
                    score_final = scores[0][i]
                    maximum_height = (bottom - top)
                    last_index = i
                else:
                    if last_index == -1:
                        continue
                    (left_out, right_out, top_out, bottom_out) = (boxes[0][last_index][1] * im_width,
                                                                  boxes[0][last_index][3] * im_width,
                                                                  boxes[0][last_index][0] * im_height,
                                                                  boxes[0][last_index][2] * im_height)

                    if left_out <= left and \
                            right_out >= right and \
                            classes[0][i] == 2 and \
                            (bottom_out - top_out) == maximum_height and \
                            left - left_out <= (right_out - left_out) * 2 / 100:
                        box_final = [int(round(left_out)), int(round(right_out)), int(round(top_out)),
                                     int(round(bottom_out))]
                        class_final = classes[0][i]
                        score_final = scores[0][i]
                        maximum_height = (bottom - top)

        if not (box_final is None):
            self.resize_box(image, box_final)
        return [box_final, image], class_final, score_final
