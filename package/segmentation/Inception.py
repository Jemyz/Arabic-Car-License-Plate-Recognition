import os

import cv2
import numpy as np
import tensorflow as tf

from package.segmentation.neural_network import binary
from package.segmentation.neural_network import label_map_util
from package.segmentation.neural_network import visualization_utils as vis_util
from package.segmentation.segmentaion_abstract import SegmentationAbstract


class Inception(SegmentationAbstract):

    def __init__(self):
        # Path to image
        # PATH_TO_IMAGE = os.path.join(CWD_PATH, IMAGE_NAME)
        print("loading segmentation model")
        # Name of the directory containing the object detection module we're using
        model_name = 'inference_graph'

        # Grab path to current working directory
        self.CWD_PATH = os.getcwd()

        # Path to frozen detection graph .pb file, which contains the model that is used
        # for object detection.
        path_to_model = os.path.join(self.CWD_PATH, "package", "segmentation", "neural_network", model_name,
                                     'frozen_inference_graph.pb')

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

    def find(self, image):
        # Perform the actual detection by running the model with the image as input
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})

        xs = np.argsort(boxes[0, :, 1])
        boxes_final = []
        classes_final = []
        scores_final = []

        for i in xs:
            if scores[0][i] > 0.8:
                boxes_final.append(boxes[0][i])
                classes_final.append(classes[0][i])
                scores_final.append(scores[0][i])

        images = binary.binarize(image, boxes_final, classes_final, scores_final)
        return images, boxes_final, classes_final, scores_final

    def visualize(self, image, directory, boxes, classes, scores, num_classes):
        # Number of classes the object detector can identify
        path_to_label = os.path.join(self.CWD_PATH, "package", "segmentation", "neural_network", 'labelmap.pbtxt')

        # Load the label map.
        # Label maps map indices to category names, so that when our convolution
        # network predicts `5`, we know that this corresponds to `king`.
        # Here we use internal utility functions, but anything that returns a
        # dictionary mapping integers to appropriate string labels would be fine
        label_map = label_map_util.load_labelmap(path_to_label)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.7)
        cv2.imwrite(directory, image)
        return image
