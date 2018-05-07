# -*- coding: utf-8 -*-

import cv2
import numpy as np
import glob
import time
from package.classifiers.classification_abstract import ClassificationAbstract
import os
from scipy.misc import imread


class TemplateMatching(ClassificationAbstract):

    def __init__(self):
        method_number = 0

        self.model_dir = os.path.join(os.getcwd(), "package", "classifiers", "template_matching/")

        self.width = 28
        self.height = 28

        self.plates_numbers = []
        self.number_of_wrong_symbols_in_plates_numbers1 = {}
        self.number_of_wrong_symbols_in_plates_numbers2 = {}
        self.number_of_wrong_symbols_in_plates_numbers3 = {}
        self.number_of_wrong_symbols_in_plates_numbers4 = {}
        self.number_of_wrong_symbols_in_plates_numbers5 = {}
        self.number_of_wrong_symbols_in_plates_numbers6 = {}

        # All the 6 methods for comparison in a list
        self.methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED', 'cv2.TM_CCORR',
                        'cv2.TM_CCORR_NORMED', 'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
        self.number_of_methods = len(self.methods)
        self.method_number = method_number

        self.true_positive = [0] * self.number_of_methods

        self.dataset_letters = []
        self.dataset_numbers = []
        self.value_letters = []
        self.value_numbers = []

        self.loadAllData("numbers")
        self.loadAllData("letters")

    def loadData(self, type):

        temp_dataset = []
        temp_value = []

        if (type == "letters"):
            self.dataset_path = self.model_dir + "plates_" + type + "/dataset/"
            self.testset_path = self.model_dir + "plates_" + type + "/testset/"

        elif (type == "numbers"):
            self.dataset_path = self.model_dir + "plates_" + type + "/dataset/"
            self.testset_path = self.model_dir + "plates_" + type + "/testset/"

        import os

        count = 0

        for subdir, dirs, files in os.walk(self.dataset_path):
            for dir in dirs:
                for subdir, dirs, files in os.walk(self.dataset_path + "/" + dir):
                    for file in files:
                        count = count + 1
                        if (count == 100):
                            break
                        imgPath = self.dataset_path + "/" + dir + "/" + file
                        image = cv2.imread(imgPath, 0)
                        image = cv2.resize(image, (self.width, self.height))
                        temp_dataset.append(image)
                        temp_value.append((imgPath[-5]))

                        import random
                        random.shuffle(files)

                    count = 0

        if (type == "letters"):

            self.dataset_letters = temp_dataset
            self.value_letters = temp_value

        elif (type == "numbers"):

            self.dataset_numbers = temp_dataset
            self.value_numbers = temp_value

    def loadAllData(self, type):

        temp_dataset = []
        temp_value = []

        if (type == "letters"):
            self.dataset_path = self.model_dir + "plates_" + type + "/dataset/"
            self.testset_path = self.model_dir + "plates_" + type + "/testset/"

        elif (type == "numbers"):
            self.dataset_path = self.model_dir + "plates_" + type + "/dataset/"
            self.testset_path = self.model_dir + "plates_" + type + "/testset/"

        for imgPath in glob.glob(self.dataset_path + '/**/*.*'):
            try:
                image = imread(imgPath)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                image = np.asarray(image)
                image = cv2.resize(image, (self.width, self.height))
                temp_dataset.append(image)
                temp_value.append((imgPath[-5]))
            except:
                ""

        if (type == "letters"):
            self.dataset_letters = temp_dataset
            self.value_letters = temp_value

        elif (type == "numbers"):

            self.dataset_numbers = temp_dataset
            self.value_numbers = temp_value

    def validate(self, type):

        start_time = time.time()

        dataset = []
        value = []

        if (type == "letters"):

            dataset = self.dataset_letters
            value = self.value_letters

        elif (type == "numbers"):

            dataset = self.dataset_numbers
            value = self.value_numbers

        number_of_imgs_in_testset = 0
        for imgPath in glob.glob(self.testset_path + '/**/*.*'):

            if (number_of_imgs_in_testset == 30):
                break

            number_of_imgs_in_testset += 1
            test_image = cv2.imread(imgPath, 0)
            test_image = cv2.resize(test_image, (self.width, self.height))

            min_percentages_of_acceptance = []
            max_percentages_of_acceptance = []

            self.plates_numbers.append((((imgPath).split("/")[-1]).split("-")[0]))

            self.number_of_wrong_symbols_in_plates_numbers1[self.plates_numbers[-1]] = 0
            self.number_of_wrong_symbols_in_plates_numbers2[self.plates_numbers[-1]] = 0
            self.number_of_wrong_symbols_in_plates_numbers3[self.plates_numbers[-1]] = 0
            self.number_of_wrong_symbols_in_plates_numbers4[self.plates_numbers[-1]] = 0
            self.number_of_wrong_symbols_in_plates_numbers5[self.plates_numbers[-1]] = 0
            self.number_of_wrong_symbols_in_plates_numbers6[self.plates_numbers[-1]] = 0

            for image in dataset:
                min_method_acceptance = np.array([])
                max_method_acceptance = np.array([])
                for meth in self.methods:
                    method = eval(meth)

                    # Apply template Matching
                    res = cv2.matchTemplate(image, test_image, method)
                    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
                    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                        min_method_acceptance = np.append(min_method_acceptance, min_val)
                    else:
                        max_method_acceptance = np.append(max_method_acceptance, max_val)

                min_percentages_of_acceptance.append(min_method_acceptance)
                max_percentages_of_acceptance.append(max_method_acceptance)

            min_percentages_of_acceptance = np.array(min_percentages_of_acceptance)
            max_percentages_of_acceptance = np.array(max_percentages_of_acceptance)

            min_value_detected_index = min_percentages_of_acceptance.argmin(axis=0)
            max_value_detected_index = max_percentages_of_acceptance.argmax(axis=0)

            result_numbers = np.append(max_value_detected_index, min_value_detected_index)

            for j in range(len(result_numbers)):
                if value[result_numbers[j]] == (imgPath[-5]):
                    self.true_positive[j] = self.true_positive[j] + 1
                else:
                    if (j == 0):
                        self.number_of_wrong_symbols_in_plates_numbers1[self.plates_numbers[-1]] += 1
                    elif (j == 1):
                        self.number_of_wrong_symbols_in_plates_numbers2[self.plates_numbers[-1]] += 1
                    elif (j == 2):
                        self.number_of_wrong_symbols_in_plates_numbers3[self.plates_numbers[-1]] += 1
                    elif (j == 3):
                        self.number_of_wrong_symbols_in_plates_numbers4[self.plates_numbers[-1]] += 1
                    elif (j == 4):
                        self.number_of_wrong_symbols_in_plates_numbers5[self.plates_numbers[-1]] += 1
                    else:
                        self.number_of_wrong_symbols_in_plates_numbers6[self.plates_numbers[-1]] += 1

        acc = []
        naa = []

        number_of_plates = len(self.number_of_wrong_symbols_in_plates_numbers1)
        number_of_wrong_plates = len(
            dict((k, v) for k, v in self.number_of_wrong_symbols_in_plates_numbers1.items() if v > 0))
        plate_accuracy = 1 - number_of_wrong_plates * 1.0 / number_of_plates
        acc.append(plate_accuracy)
        naa.append(number_of_plates - number_of_wrong_plates)

        number_of_plates = len(self.number_of_wrong_symbols_in_plates_numbers2)
        number_of_wrong_plates = len(
            dict((k, v) for k, v in self.number_of_wrong_symbols_in_plates_numbers2.items() if v > 0))
        plate_accuracy = 1 - number_of_wrong_plates * 1.0 / number_of_plates
        acc.append(plate_accuracy)
        naa.append(number_of_plates - number_of_wrong_plates)

        number_of_plates = len(self.number_of_wrong_symbols_in_plates_numbers3)
        number_of_wrong_plates = len(
            dict((k, v) for k, v in self.number_of_wrong_symbols_in_plates_numbers3.items() if v > 0))
        plate_accuracy = 1 - number_of_wrong_plates * 1.0 / number_of_plates
        acc.append(plate_accuracy)
        naa.append(number_of_plates - number_of_wrong_plates)

        number_of_plates = len(self.number_of_wrong_symbols_in_plates_numbers4)
        number_of_wrong_plates = len(
            dict((k, v) for k, v in self.number_of_wrong_symbols_in_plates_numbers4.items() if v > 0))
        plate_accuracy = 1 - number_of_wrong_plates * 1.0 / number_of_plates
        acc.append(plate_accuracy)
        naa.append(number_of_plates - number_of_wrong_plates)

        number_of_plates = len(self.number_of_wrong_symbols_in_plates_numbers5)
        number_of_wrong_plates = len(
            dict((k, v) for k, v in self.number_of_wrong_symbols_in_plates_numbers5.items() if v > 0))
        plate_accuracy = 1 - number_of_wrong_plates * 1.0 / number_of_plates
        acc.append(plate_accuracy)
        naa.append(number_of_plates - number_of_wrong_plates)

        number_of_plates = len(self.number_of_wrong_symbols_in_plates_numbers6)
        number_of_wrong_plates = len(
            dict((k, v) for k, v in self.number_of_wrong_symbols_in_plates_numbers6.items() if v > 0))
        plate_accuracy = 1 - number_of_wrong_plates * 1.0 / number_of_plates
        acc.append(plate_accuracy)
        naa.append(number_of_plates - number_of_wrong_plates)

        end_time = time.time()

        accuracy = [x / (number_of_imgs_in_testset * 1.0) for x in self.true_positive]

        print("True Positive = ", self.true_positive)
        print("Total Number of Testset = ", number_of_imgs_in_testset)
        print("accuracy = ", accuracy)
        print("True Positive of Plates = ", naa)
        print("Total Number of Plates = ", number_of_plates)
        print("plate accuracy = ", acc)
        print("time = ", (end_time - start_time), "sec")

    def predict(self, image, type):

        dataset = []
        value = []

        if type == 1:
            dataset = self.dataset_letters
            value = self.value_letters

        elif type == 2:
            dataset = self.dataset_numbers
            value = self.value_numbers

        test_image = cv2.resize(image, (self.width, self.height))

        min_percentages_of_acceptance = []
        max_percentages_of_acceptance = []
        method = eval(self.methods[self.method_number])

        for image in dataset:
            min_method_acceptance = np.array([])
            max_method_acceptance = np.array([])

            # for meth in self.methods:

            # Apply template Matching
            res = cv2.matchTemplate(image, test_image, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                min_method_acceptance = np.append(min_method_acceptance, min_val)
                min_percentages_of_acceptance.append(min_method_acceptance)

            else:
                max_method_acceptance = np.append(max_method_acceptance, max_val)
                max_percentages_of_acceptance.append(max_method_acceptance)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            min_percentages_of_acceptance = np.array(min_percentages_of_acceptance)
            min_value_detected_index = min_percentages_of_acceptance.argmin(axis=0)
            result_numbers = min_value_detected_index


        else:
            max_percentages_of_acceptance = np.array(max_percentages_of_acceptance)
            max_value_detected_index = max_percentages_of_acceptance.argmax(axis=0)
            result_numbers = max_value_detected_index

        # for j in range(len(result_numbers)):
        #    print(self.value[result_numbers[j]])

        return value[result_numbers[0]], 1
