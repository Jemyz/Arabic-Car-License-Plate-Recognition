import abc


class ClassificationAbstract(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict(self, image, type):
        """Required Method"""
        return