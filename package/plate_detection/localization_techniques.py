import cv2
from package.plate_detection.localization_abstract import LocalizationAbstract
from threading import Semaphore
from package.plate_detection.detect_plate import PlateDetection
from package.plate_detection.object_detection_plate import ObjectDetection
from package.plate_detection.classify import classify

model_map = {
    "Inception": ObjectDetection,
    "ResNet101": ObjectDetection,
    "Inception-ResNet": ObjectDetection
}


class Localize(object):
    def __init__(self, strategies=None):
        self.__action = {}
        self.__semaphores = {}

        self.count = 0
        self.set_strategy(strategies)

    def set_strategy(self, strategies):

        if not strategies:
            strategies = {"PlateDetection": 1}

        for strategy in strategies:
            size = strategies[strategy]

            if strategy in globals():
                strategy = globals()[strategy]

            if size == 0:
                raise ValueError

            if strategy in model_map or issubclass(strategy, LocalizationAbstract):
                if strategy in model_map:
                    localization_strategy = model_map[strategy](strategy)
                else:
                    localization_strategy = strategy()

                self.__action[strategy] = [localization_strategy for _ in range(size)]
                self.__semaphores[strategy] = Semaphore(value=size)
            else:
                raise TypeError

    def localize(self, image, localization_strategy=ObjectDetection, get_object=False, localization_object=None,
                 load_model=False):
        try:

            if not localization_object:
                if load_model:

                    if localization_strategy in model_map:
                        localization_object = model_map[localization_strategy](localization_strategy)
                    else:
                        localization_object = localization_strategy()
                else:
                    localization_object = self.acquire_localization_strategy(localization_strategy)

            value_array = localization_object.find(image)

            if not get_object and not load_model:
                self.append_localization_strategy(localization_strategy, localization_object)
                return value_array, ""

            return value_array, localization_object

        except Exception as e:

            print('%s (%s)' % (e, type(e)))
            if not get_object and not load_model:
                self.append_localization_strategy(localization_strategy, localization_object)

    def visualize(self, image, directory, box, class_detected):
        print(box)
        print(int(class_detected))

        if int(class_detected) == 1:
            value_array = cv2.rectangle(image, (box[0], box[3]), (box[1], box[2]), (0, 0, 255), 6)
        else:
            value_array = cv2.rectangle(image, (box[0], box[3]), (box[1], box[2]), (255, 0, 0), 6)

        cv2.imwrite(directory, value_array)

        return value_array

    def acquire_localization_strategy(self, localization_strategy):
        self.__semaphores[localization_strategy].acquire()
        return self.__action[localization_strategy].pop()

    def append_localization_strategy(self, localization_strategy, localization_object):
        self.__action[localization_strategy].append(localization_object)
        self.__semaphores[localization_strategy].release()
