import tensorflow as tf
import numpy as np
from package.plate_detection.misc import box_area, read_image
from scipy.misc import imshow
import argparse
import os

PATH_TO_PLATE_OR_NON_PLATE_CLASSIFIER = os.path.join(os.getcwd(), "package", "plate_detection",
                                                     "plate_non_plate_classifier", "model",
                                                     'retrained_graph.pb')
PATH_TO_PLATE_OR_NON_PLATE_LABELS = os.path.join(os.getcwd(), "package", "plate_detection",
                                                 "plate_non_plate_classifier", "data",
                                                 'retrained_labels.txt')
PLATE_NON_PLATE_LABELS = [line.rstrip() for line in tf.gfile.GFile(PATH_TO_PLATE_OR_NON_PLATE_LABELS)]


class PlateClassifier(object):
    def __init__(self):
        with tf.gfile.FastGFile(PATH_TO_PLATE_OR_NON_PLATE_CLASSIFIER, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())

    def load_graph(self):
        tf.reset_default_graph()
        tf.import_graph_def(self.graph_def, name='pnp')

    def get_plates(self, boxes, img):
        img = read_image(img)
        self.load_graph()
        with tf.Session() as sess:
            soft_max_tensor = sess.graph.get_tensor_by_name('pnp/final_result:0')
            predictions = [
                sess.run(
                    soft_max_tensor,
                    {'pnp/DecodeJpeg:0': img[box[2]:box[3], box[0]:box[1]]}
                ) for box in boxes
            ]
            # sorting data based on confidence
            sorted_labels = [prediction[0].argsort()[-len(prediction[0]):][::-1] for prediction in predictions]
            labels = np.asarray([label[0] for label in sorted_labels])
            plates_args = np.where(labels == 1)
            plates_boxes = boxes[plates_args[0].tolist()]
            return plates_boxes, img

    def get_largest_plate(self, boxes, img):
        img = read_image(img)
        boxes = sorted(boxes, key=box_area, reverse=True)
        self.load_graph()
        with tf.Session() as sess:
            soft_max_tensor = sess.graph.get_tensor_by_name('pnp/final_result:0')
            for box in boxes:
                prediction = sess.run(
                    soft_max_tensor,
                    {'pnp/DecodeJpeg:0': img[box[2]:box[3], box[0]:box[1]]}
                )
                if prediction[0].argsort()[-len(prediction[0]):][::-1][0] == 1:
                    return box, img
        return None, img

    def classify(self, img):
        img = read_image(img)
        self.load_graph()
        with tf.Session() as sess:
            soft_max_tensor = sess.graph.get_tensor_by_name('pnp/final_result:0')
            predictions = sess.run(soft_max_tensor, {'pnp/DecodeJpeg:0': img})
            # Sort to show data of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            for node_id in top_k:
                human_string = PLATE_NON_PLATE_LABELS[node_id]
                score = predictions[0][node_id]
                print('%s (score = %.5f)' % (human_string, score))
        return img


def run(args):
    classifier = PlateClassifier()
    img = classifier.classify(args.image)
    if args.view:
        imshow(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', '--img',
        type=str,
        help='Path to vehicle image.'
    )
    parser.add_argument(
        '--view',
        type=bool,
        default=False,
        help='Whether to view plate image or not.'
    )
    run(parser.parse_args())
