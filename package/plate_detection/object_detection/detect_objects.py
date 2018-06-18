import argparse
import numpy as np
import tensorflow as tf
from scipy.misc import imshow
from package.plate_detection.object_detection.utils import ops as utils_ops
from package.plate_detection.object_detection.utils import label_map_util
from package.plate_detection.object_detection.utils import visualization_utils as vis_util
from package.plate_detection.misc import read_image
import os

PATH_TO_MODEL = os.path.join(os.getcwd(), "package", "plate_detection",
                             "object_detection", "model",
                             'frozen_inference_graph.pb')

PATH_TO_LABELS = os.path.join(os.getcwd(), "package", "plate_detection",
                              "object_detection", "data",
                              'mscoco_label_map.pbtxt')
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

    def get_objects(self, img, view=False):
        img = read_image(img)
        output_dict = self._detect_objects(img)
        if view:
            show(
                img=img.copy(),
                boxes=output_dict['detection_boxes'],
                labels=output_dict['detection_classes'],
                scores=output_dict['detection_scores'],
                masks=output_dict.get('detection_masks'),
                thresh=self.thresh
            )
        # filtering by classes
        if self.classes is not None:
            class_filtered = np.isin(output_dict['detection_classes'], self.classes)
            print(class_filtered)
            output_dict['detection_scores'] = output_dict['detection_scores'][class_filtered]
            output_dict['detection_classes'] = output_dict['detection_classes'][class_filtered]
            output_dict['detection_boxes'] = output_dict['detection_boxes'][class_filtered]
        # filtering by scores
        scores_filtered = np.where(output_dict['detection_scores'] > self.thresh)
        return output_dict['detection_classes'][scores_filtered], output_dict['detection_scores'][scores_filtered], \
            output_dict['detection_boxes'][scores_filtered], output_dict.get('detection_masks'), img

    def get_largest_object(self, img, view=False):
        """return the object with largest area"""
        labels, scores, boxes, masks, img = self.get_objects(img, view)
        if boxes.size > 0:
            # if any of required classes exist, return the largest one.
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            max_arg = np.argmax(areas)
            return labels[max_arg], scores[max_arg], boxes[max_arg], masks, img
        return None, None, None, None, img


def show(img, boxes, labels, scores, masks, thresh):
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        boxes,
        labels,
        scores,
        category_index,
        instance_masks=masks,
        use_normalized_coordinates=True,
        line_thickness=2,
        min_score_thresh=thresh
    )
    imshow(img)


def run(args):
    od = ObjectDetection(args.thresh, args.classes)
    labels, scores, boxes, masks, image = od.get_largest_object(args.image)
    if args.view:
        show(
            img=image.copy(),
            boxes=boxes,
            labels=labels,
            scores=scores,
            masks=masks,
            thresh=args.thresh
        )


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
