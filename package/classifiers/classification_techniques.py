from package.classifiers.cnn import CNN
from threading import Semaphore
from package.classifiers.classification_abstract import ClassificationAbstract
from package.classifiers.svm import SVM
from package.classifiers.imagenet import ImageNet
from package.classifiers.template_matching import TemplateMatching
from package.classifiers.inception import Inception
from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import Xception
from keras.applications import InceptionV3
from keras.applications import InceptionResNetV2
from keras.applications import NASNetLarge
from keras.applications import ResNet50
from keras.applications import DenseNet201
from keras.applications import DenseNet169
from keras.applications import DenseNet121
from keras.applications import MobileNet  # 128
from keras.applications import NASNetMobile  # 224

model_map = {MobileNet: ImageNet, NASNetMobile: ImageNet,
             DenseNet121: ImageNet, DenseNet169: ImageNet,
             DenseNet201: ImageNet, ResNet50: ImageNet,
             NASNetLarge: ImageNet, Xception: ImageNet,
             InceptionV3: ImageNet, InceptionResNetV2: ImageNet,
             VGG16: ImageNet, VGG19: ImageNet}


class Classifier(object):
    def __init__(self, strategies=None):
        self.__action = {}
        self.__semaphores = {}
        self.both = False
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

            if strategy in model_map or issubclass(strategy, ClassificationAbstract):
                if strategy in model_map:
                    classification_object = model_map[strategy](strategy, self.both)
                else:
                    classification_object = strategy()

                self.__action[strategy] = [classification_object for _ in range(size)]
                self.__semaphores[strategy] = Semaphore(value=size)
            else:
                raise TypeError

    def classify(self, image, type_image, classification_strategy=CNN, get_object=False, classification_object=None,
                 load_model=False):
        try:

            if classification_object is None:
                if load_model:

                    if classification_strategy in model_map:
                        classification_object = model_map[classification_strategy](classification_strategy, self.both)
                    else:
                        classification_object = classification_strategy()
                else:
                    classification_object = self.acquire_classification_strategy(classification_strategy)

            value_array = classification_object.predict(image, type_image)

            if not get_object and not load_model:
                self.append_classification_strategy(classification_strategy, classification_object)
                return value_array

            return value_array, classification_object

        except Exception as e:

            print('%s (%s)' % (e, type(e)))
            if not get_object and not load_model:
                self.append_classification_strategy(classification_strategy, classification_object)

    def acquire_classification_strategy(self, classification_strategy):
        print("classification lock:value")
        print(self.__semaphores[classification_strategy]._value)
        self.__semaphores[classification_strategy].acquire()
        return self.__action[classification_strategy].pop()

    def append_classification_strategy(self, classification_strategy, classification_object):
        self.__action[classification_strategy].append(classification_object)
        self.__semaphores[classification_strategy].release()
