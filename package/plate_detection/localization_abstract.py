import abc


class LocalizationAbstract(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def find(self, image):
        """Required Method"""
        return