import abc


class SegmentationAbstract(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def find(self, image):
        """Required Method"""
        return

    def visualize(self, image, directory, boxes, classes, scores, num_classes):
        return
