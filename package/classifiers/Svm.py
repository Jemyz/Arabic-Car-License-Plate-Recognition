# -*- coding: utf-8 -*-

# from deep_learning.accuracy import Accuracy

import glob
from sklearn import svm
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import time
import pickle
from package.classifiers.classification_abstract import ClassificationAbstract
import os


class Svm(ClassificationAbstract):

    def __init__(self):

        self.width = 28
        self.height = 28

        self.datasetdata = []
        self.dataset_targets = []

        self.testset = []
        self.testdata = []
        self.testset_targets = []

        self.orignal_images = []
        self.plates_numbers = []
        self.number_of_wrong_symbols_in_plates_numbers = {}

        self.dataset_path = ""
        self.testset_path = ""
        self.type = ""

        filename = 'svm_char_50_linear'
        self.path_to_model = os.path.join(os.getcwd(), "package", "classifiers", "svm", filename)
        # load the model from disk
        loaded_model = pickle.load(open(self.path_to_model, 'rb'))
        self.clfChar = loaded_model

        # load the model from disk
        filename = 'svm_num_50_linear'
        self.path_to_model = os.path.join(os.getcwd(), "package", "classifiers", "svm", filename)
        loaded_model = pickle.load(open(self.path_to_model, 'rb'))
        self.clfNum = loaded_model

    def trainModel(self, type, C=50, kernel="linear"):

        self.type = type
        if (type == "letters"):
            self.dataset_path = "plates_" + type + "/dataset/"
            self.testset_path = "plates_" + type + "/testset/"
        elif (type == "numbers"):
            self.dataset_path = "plates_" + type + "/dataset/"
            self.testset_path = "plates_" + type + "/testset/"

        self.datasetdata = []
        self.dataset_targets = []
        for imgPath in glob.glob(self.dataset_path + '/**/*.*'):
            image = cv2.imread(imgPath, 0)
            image = 255 - image
            image[image < 128] = 0
            image[image >= 128] = 255

            image = cv2.resize(image, (self.width, self.height))
            image = np.array(image, dtype=np.float64).ravel()
            reshaped_image = image.reshape(1, -1)

            scaler = Normalizer().fit(reshaped_image)
            rescaled_image = scaler.transform(reshaped_image)

            self.datasetdata.append(rescaled_image[0])

            self.dataset_targets.append((imgPath)[-5])

        self.clf = svm.SVC(C=50, kernel="linear")

        self.clf.fit(self.datasetdata, self.dataset_targets)

        if (type == "letters"):
            # save the model to disk
            filename = "svm_char_" + str(C) + "_" + kernel
            pickle.dump(self.clf, open(filename, 'wb'))
        elif (type == "numbers"):
            # save the model to disk
            filename = "svm_num_" + str(C) + "_" + kernel
            pickle.dump(self.clf, open(filename, 'wb'))

    def validate(self):

        if (self.type == "letters"):
            self.dataset_path = "plates_" + self.type + "/dataset/"
            self.testset_path = "plates_" + self.type + "/testset/"
        elif (self.type == "numbers"):
            self.dataset_path = "plates_" + self.type + "/dataset/"
            self.testset_path = "plates_" + self.type + "/testset/"

        start_time = time.time()

        for imgPath in glob.glob(self.testset_path + '/**/*.*'):
            test_image = cv2.imread(imgPath, 0)
            test_image = 255 - test_image
            test_image[test_image < 128] = 0
            test_image[test_image >= 128] = 255

            self.orignal_images.append(test_image)
            test_image = cv2.resize(test_image, (self.width, self.height))
            self.testset.append(test_image)
            test_image = np.array(test_image, dtype=np.float64).ravel()

            reshaped_image = test_image.reshape(1, -1)

            scaler = Normalizer().fit(reshaped_image)
            rescaled_image = scaler.transform(reshaped_image)

            self.testdata.append(rescaled_image)

            self.testset_targets.append((imgPath)[-5])

            self.plates_numbers.append((((imgPath).split("/")[-1]).split("-")[0]))
            self.number_of_wrong_symbols_in_plates_numbers[self.plates_numbers[-1]] = 0

        true_positive = 0

        for i in range(len(self.testdata)):

            if self.clf.predict(self.testdata[i])[0] == self.testset_targets[i]:
                true_positive += 1

            else:
                self.number_of_wrong_symbols_in_plates_numbers[self.plates_numbers[i]] += 1
                # print
                # print "predict: " + (clf.predict(testdata[i])[0]), " real " + testset_targets[i]
                # print
                # cv2.imshow("image", orignal_images[i])
                # cv2.waitKey()

        number_of_plates = len(self.number_of_wrong_symbols_in_plates_numbers)
        number_of_wrong_plates = len(
            dict((k, v) for k, v in self.number_of_wrong_symbols_in_plates_numbers.items() if v > 0))

        plate_accuracy = 1 - number_of_wrong_plates * 1.0 / number_of_plates
        number_of_imgs_in_testset = len(self.testdata)
        accuracy = true_positive / (number_of_imgs_in_testset * 1.0)
        end_time = time.time()

        # a = Accuracy()
        # a.calc_accuracy()

        print(self.type)
        print("True Positive = ", true_positive)
        print("Total Number of Testset = ", number_of_imgs_in_testset)
        print("accuracy = ", accuracy)
        print("True Positive of Plates = ", (number_of_plates - number_of_wrong_plates))
        print("Total Number of Plates = ", number_of_plates)

        print("plate accuracy = ", plate_accuracy)
        print("time = ", (end_time - start_time), "sec")

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_xlabel('C')
        # ax.set_ylabel('Accuracy')
        # ax.plot(Css,accurs)
        # plt.show()

    def predict(self, image, type):

        image = cv2.resize(image, (self.width, self.height))
        image = cv2.bitwise_not(image)
        test_image = np.array(image, dtype=np.float64).ravel()

        reshaped_image = test_image.reshape(1, -1)

        scalar = Normalizer().fit(reshaped_image)
        rescaled_image = scalar.transform(reshaped_image)
        result = 0

        if type == 1:
            result = self.clfChar.predict(rescaled_image)[0]
        elif type == 2:
            result = self.clfNum.predict(rescaled_image)[0]

        return result, 1
