import cv2
from package.plate_detection.localization_abstract import LocalizationAbstract
from threading import Semaphore
from package.plate_detection.detect_plate import PlateDetection


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

            if issubclass(strategy, LocalizationAbstract):
                self.__action[strategy] = [strategy() for _ in range(size)]
                self.__semaphores[strategy] = Semaphore(value=size)
            else:
                raise TypeError

    def localize(self, image, localization_strategy=PlateDetection, get_object=False, localization_object=None):

        if not localization_object:
            localization_object = self.acquire_localization_strategy(localization_strategy)

        value_array = localization_object.find(image)

        if not get_object:
            self.append_localization_strategy(localization_strategy, localization_object)
            return value_array

        return value_array, localization_object

    def visualize(self, image, directory, box, localization_strategy=PlateDetection, get_object=False,
                  localization_object=None):

        if not localization_object:
            localization_object = self.acquire_localization_strategy(localization_strategy)
        print(box)
        value_array = cv2.rectangle(image, (box[0], box[3]), (box[1], box[2]), (0, 0, 255), 4)
        cv2.imwrite(directory, value_array)

        if not get_object:
            self.append_localization_strategy(localization_strategy, localization_object)
            return value_array

        return value_array, localization_object

    def acquire_localization_strategy(self, localization_strategy):
        self.__semaphores[localization_strategy].acquire()
        return self.__action[localization_strategy].pop()

    def append_localization_strategy(self, localization_strategy, localization_object):
        self.__action[localization_strategy].append(localization_object)
        self.__semaphores[localization_strategy].release()
