from package.classifiers.cnn import CNN
from threading import Semaphore
from package.classifiers.classification_abstract import ClassificationAbstract
from package.classifiers.Svm import Svm
from package.classifiers.imagenet import ImageNet
from package.classifiers.TemplateMatching import TemplateMatching


class Classifier(object):
    def __init__(self, strategies=None):
        self.__action = {}
        self.__semaphores = {}

        self.count = 0
        self.set_strategy(strategies)

    def set_strategy(self, strategies):

        if not strategies:
            strategies = {"cnn": 1}

        for strategy in strategies:
            size = strategies[strategy]

            if strategy in globals():
                strategy = globals()[strategy]

            if size == 0:
                raise ValueError

            if issubclass(strategy, ClassificationAbstract):
                self.__action[strategy] = [strategy() for _ in range(size)]
                self.__semaphores[strategy] = Semaphore(value=size)
            else:
                raise TypeError

    def classify(self, image, type_image, classification_strategy=CNN, get_object=False, classification_object=None):

        if not classification_object:
            classification_object = self.acquire_classification_strategy(classification_strategy)

        value_array = classification_object.predict(image, type_image)

        if not get_object:
            self.append_classification_strategy(classification_strategy, classification_object)
            return value_array

        return value_array, classification_object

    def acquire_classification_strategy(self, classification_strategy):
        print("classification lock:value")
        print(self.__semaphores[classification_strategy]._value)
        self.__semaphores[classification_strategy].acquire()
        return self.__action[classification_strategy].pop()

    def append_classification_strategy(self, classification_strategy, classification_object):
        self.__action[classification_strategy].append(classification_object)
        self.__semaphores[classification_strategy].release()
