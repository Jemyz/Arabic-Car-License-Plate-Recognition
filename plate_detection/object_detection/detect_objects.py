import argparse
import numpy as np
import tensorflow as tf
from scipy.misc import imshow
from plate_detection.object_detection.utils import ops as utils_ops
from plate_detection.object_detection.utils import label_map_util
from plate_detection.object_detection.utils import visualization_utils as vis_util
from plate_detection.misc import read_image

PATH_TO_LABELS = 'object_detection/data/mscoco_label_map.pbtxt'
PATH_TO_MODEL = 'object_detection/model/frozen_inference_graph.pb'
NUM_CLASSES = 90
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


class ObjectDetection(object):
    def __init__(self, thresh, classes=None):
        with tf.gfile.GFile(PATH_TO_MODEL, 'rb') as f:
            self.graph_def = tf.GraphDef()
            self.graph_def.ParseFromString(f.read())
        self.thresh = thresh
        self.classes = classes

    def load_graph(self):
        tf.reset_default_graph()
        tf.import_graph_def(self.graph_def, name='')

    def _detect_objects(self, img):
        self.load_graph()
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Re-frame is required to translate mask from box coordinates to image
                # coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, img.shape[0], img.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(img, 0)})
            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    def get_objects(self, img):
        img = read_image(img)
        output_dict = self._detect_objects(img)
        # filtering by classes
        if self.classes is not None:
            class_filtered = np.isin(output_dict['detection_classes'], self.classes)
        # filtering by scores
        scores = output_dict['detection_scores'][class_filtered] if self.classes else output_dict['detection_scores']
        scores_filtered = np.where(scores > self.thresh)
        labels = output_dict['detection_classes'][scores_filtered]
        scores = output_dict['detection_scores'][scores_filtered]
        boxes = output_dict['detection_boxes'][scores_filtered]
        masks = output_dict.get('detection_masks')
        return labels, scores, boxes, masks, img

    def get_largest_object(self, img):
        """return the object with largest area"""
        labels, scores, boxes, masks, img = self.get_objects(img)
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        max_arg = np.argmax(areas)
        return labels[max_arg], scores[max_arg], boxes[max_arg], masks, img


def run(args):
    od = ObjectDetection(args.thresh, args.classes)
    labels, scores, boxes, masks, image = od.get_largest_object(args.image)
    if args.view:
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            boxes,
            labels,
            scores,
            category_index,
            instance_masks=masks,
            use_normalized_coordinates=True,
            line_thickness=2,
            min_score_thresh=args.thresh
        )
        imshow(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', '--img',
        type=str,
        required=True,
        help='Path to vehicle image.'
    )
    parser.add_argument(
        '--classes', '-c',
        type=list,
        default=None,
        help='Classes to be detected.'
    )
    parser.add_argument(
        '--thresh',
        type=float,
        default=0.5,
        help='Min score accepted.'
    )
    parser.add_argument(
        '--view',
        type=bool,
        default=True,
        help='Whether to view plate image or only save it.'
    )
    run(parser.parse_args())
