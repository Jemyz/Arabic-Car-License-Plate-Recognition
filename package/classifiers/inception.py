from package.classifiers.classification_abstract import ClassificationAbstract
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img
import cv2


class Inception(ClassificationAbstract):
    def __init__(self):
        self.model_dir =  os.path.join(os.getcwd(),
                                      "package",
                                      "classifiers",
                                      "models",
                                      "Inception",
                                      "stable/"
                                      )
        self.letters_model_dir = self.model_dir + "letters.pb"
        self.letters_labels_dir = self.model_dir + "letters.txt"

        self.numbers_model_dir = self.model_dir + "numbers.pb"
        self.numbers_labels_dir = self.model_dir + "numbers.txt"

        self.letters_labels = [line.lstrip() for line in tf.gfile.GFile(self.letters_labels_dir)]
        self.numbers_labels = [line.rstrip() for line in tf.gfile.GFile(self.numbers_labels_dir)]


        with tf.gfile.FastGFile(self.letters_model_dir, 'rb') as f:
            self.letters_graph_def = tf.GraphDef()
            self.letters_graph_def.ParseFromString(f.read())

        with tf.gfile.FastGFile(self.numbers_model_dir, 'rb') as f:
            self.numbers_graph_def = tf.GraphDef()
            self.numbers_graph_def.ParseFromString(f.read())


    def load_graph(self,type):
        tf.reset_default_graph()
        if(type == 1):
            tf.import_graph_def(self.letters_graph_def, name='pnp')
        else:
            tf.import_graph_def(self.numbers_graph_def,name='pnp')


    def predict(self, image, type):
        self.load_graph(type)

        image = cv2.resize(image, (48, 48))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        with tf.Session() as sess:
            soft_max_tensor = sess.graph.get_tensor_by_name('pnp/final_result:0')
            predictions = sess.run(soft_max_tensor, {'pnp/DecodeJpeg:0': image})
            # Sort to show data of first prediction in order of confidence
            top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
            if(type == 1):
                pred_label = self.letters_labels[top_k[0]]
            else:
                pred_label = self.numbers_labels[top_k[0]]
            probability = predictions[0][top_k[0]]
        return pred_label,probability



    def validate(self,validation_generator, model, type, visulaize):

        if type == "letters":
            num_classes = 17
        else:
            num_classes = 10

        fnames = validation_generator.filenames
        ground_truth = validation_generator.classes
        label2index = validation_generator.class_indices
        nVal = len(ground_truth)

        # Getting the mapping from class index to class label
        idx2label = dict((v, k) for k, v in label2index.items())
        predictions = []
        prob = []
        truth = []

        for image, label in validation_generator:
            result = model.classify(image[0])
            predictions.append(str(result[0]))
            prob.append(result[1])
            truth.append(str(label[0]))

        predictions = np.array(predictions)
        prob = np.array(predictions)
        truth = np.array(truth)

        errors = np.where(predictions != truth)[0]
        print("No of errors = {}/{}".format(len(errors), nVal))
        print("Acc = " + str(1 - len(errors) / (nVal * 1.0)))
        if (visulaize):
            self.visualize(errors, prob, idx2label, fnames, type)
        return errors, prob, idx2label, fnames

    def visualize(self,errors, prob, idx2label, fnames, type):

        for i in range(len(errors)):
            pred_class = np.argmax(prob[errors[i]])
            pred_label = idx2label[pred_class]

            print('Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
                fnames[errors[i]].split('/')[0],
                pred_label,
                prob[errors[i]][pred_class]))
            validation_dir = "plates_numbers/"
            original = load_img('{}/{}'.format(validation_dir, fnames[errors[i]]))
            plt.imshow(original)
            plt.show()


