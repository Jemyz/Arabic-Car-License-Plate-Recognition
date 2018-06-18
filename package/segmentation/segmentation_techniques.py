from package.segmentation.segmentaion_abstract import SegmentationAbstract
from package.segmentation.Inception import Inception
from threading import Semaphore
import cv2

model_map = {
    "Inception": Inception,
    "ResNet101":Inception,
    "Inception-ResNet": Inception,
    "FasterRCNN-ResNet": Inception,

}


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

            if strategy in model_map or issubclass(strategy, SegmentationAbstract):
                if strategy in model_map:
                    segmentation_strategy = model_map[strategy](strategy)
                else:
                    segmentation_strategy = strategy()

                self.__action[strategy] = [segmentation_strategy for _ in range(size)]
                self.__semaphores[strategy] = Semaphore(value=size)
            else:
                raise TypeError

    def segment(self, image, segmentation_strategy=Inception, get_object=False, segmentation_object=None,
                load_model=False):

        if segmentation_object is None:
            if load_model:

                if segmentation_strategy in model_map:
                    segmentation_object = model_map[segmentation_strategy](segmentation_strategy)
                else:
                    segmentation_object = segmentation_strategy()

            else:
                segmentation_object = self.acquire_segmentation_strategy(segmentation_strategy)

        value_array = segmentation_object.find(image)
        print(value_array)

        if not get_object and not load_model:
            self.append_segmentation_strategy(segmentation_strategy, segmentation_object)
            return value_array, ""

        return value_array, segmentation_object

    def visualize(self, image, directory, boxes, classes):
        im_width = image.shape[1]
        im_height = image.shape[0]

        for i in range(len(boxes)):
            (left, right, top, bottom) = (boxes[i][1] * im_width,
                                          boxes[i][3] * im_width,
                                          boxes[i][0] * im_height,
                                          boxes[i][2] * im_height)

            if int(classes[i]) == 1:
                image = cv2.rectangle(image,
                                      (int(left), int(bottom)),
                                      (int(right), int(top)),
                                      (255, 0, 0),
                                      4)
            else:
                image = cv2.rectangle(image,
                                      (int(round(left)), int(round(bottom))),
                                      (int(round(right)), int(round(top))),
                                      (0, 255, 0),
                                      4)

        cv2.imwrite(directory, image)

    def acquire_segmentation_strategy(self, segmentation_strategy):
        print("lock:value")
        print(self.__semaphores[segmentation_strategy]._value)
        self.__semaphores[segmentation_strategy].acquire()
        return self.__action[segmentation_strategy].pop()

    def append_segmentation_strategy(self, segmentation_strategy, segmentation_object):
        self.__action[segmentation_strategy].append(segmentation_object)
        self.__semaphores[segmentation_strategy].release()
