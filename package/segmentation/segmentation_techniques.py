from package.segmentation.segmentaion_abstract import SegmentationAbstract
from package.segmentation.Inception import Inception
from threading import Semaphore


class Segmenter(object):
    def __init__(self, strategies=None):
        self.__action = {}
        self.__semaphores = {}

        self.count = 0
        self.set_strategy(strategies)

    def set_strategy(self, strategies):

        if not strategies:
            strategies = {"Inception": 1}

        for strategy in strategies:
            size = strategies[strategy]

            if strategy in globals():
                strategy = globals()[strategy]

            if size == 0:
                raise ValueError

            if issubclass(strategy, SegmentationAbstract):
                self.__action[strategy] = [strategy() for _ in range(size)]
                self.__semaphores[strategy] = Semaphore(value=size)
            else:
                raise TypeError

    def segment(self, image, segmentation_strategy=Inception, get_object=False, segmentation_object=None):

        if not segmentation_object:
            segmentation_object = self.acquire_segmentation_strategy(segmentation_strategy)

        value_array = segmentation_object.find(image)

        if not get_object:
            self.append_segmentation_strategy(segmentation_strategy, segmentation_object)
            return value_array

        return value_array, segmentation_object

    def visualize(self, image, directory, boxes=None, classes=None, scores=None, num_classes=2,
                  segmentation_strategy=Inception, get_object=False, segmentation_object=None):

        if not segmentation_object:
            segmentation_object = self.acquire_segmentation_strategy(segmentation_strategy)

        value_array = segmentation_object.visualize(image, directory, boxes, classes, scores, num_classes)

        if not get_object:
            self.append_segmentation_strategy(segmentation_strategy, segmentation_object)
            return value_array

        return value_array, segmentation_object

    def acquire_segmentation_strategy(self, segmentation_strategy):
        print("lock:value")
        print(self.__semaphores[segmentation_strategy]._value)
        self.__semaphores[segmentation_strategy].acquire()
        return self.__action[segmentation_strategy].pop()

    def append_segmentation_strategy(self, segmentation_strategy, segmentation_object):
        self.__action[segmentation_strategy].append(segmentation_object)
        self.__semaphores[segmentation_strategy].release()
