import tensorflow as tf
from package.plate_detection.misc import read_image
from scipy.misc import imshow
import argparse
import os

path_to_model = os.path.join(os.getcwd(), "package", "plate_detection", "classifier", "model")
PATH_TO_MODEL = os.path.join(path_to_model, "output_graph_best.pb")
PATH_TO_LABELS = os.path.join(path_to_model, "output_labels.txt")

# PATH_TO_MODEL = "g7.pb"
OLD_NEW_PLATE_LABELS = [line.rstrip() for line in tf.gfile.GFile(PATH_TO_LABELS)]

# Un persists graph from file
with tf.gfile.FastGFile(PATH_TO_MODEL, 'rb') as f:
    NEW_OLD_PLATE_GRAPH_DEF = tf.GraphDef()
    NEW_OLD_PLATE_GRAPH_DEF.ParseFromString(f.read())


def load_graph():
    tf.reset_default_graph()
    _ = tf.import_graph_def(NEW_OLD_PLATE_GRAPH_DEF, name='onp')


def classify(img):
    #img = read_image(img)

    load_graph()
    with tf.Session() as sess:
        soft_max_tensor = sess.graph.get_tensor_by_name('onp/final_result:0')
        predictions = sess.run(soft_max_tensor, {'onp/DecodeJpeg:0': img})
        # Sort to show data of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        labels = []
        scores = []
        for node_id in top_k:
            labels.append(OLD_NEW_PLATE_LABELS[node_id])
            scores.append(predictions[0][node_id])
    print(scores)
    print(labels)

    if scores[0] > scores[1]:
        return labels[0], scores[0]
    else:
        return labels[1], scores[1]


def run(args):
    labels, scores, img = classify(args.image)
    for label, score in zip(labels, scores):
        print(label, score)
    if args.view:
        imshow(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image', '--img',
        type=str,
        # required=True,
        default="testing images/new1_1.jpeg",
        help='Path to vehicle image.'
    )
    parser.add_argument(
        '--view',
        type=bool,
        default=False,
        help='Whether to view plate image or not.'
    )
    run(parser.parse_args())
