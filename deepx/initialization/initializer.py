import six
from abc import ABCMeta, abstractmethod

@six.add_metaclass(ABCMeta)
class Initializer(object):

    def __init__(self, shape, dtype=None, seed=None):
        self.shape = shape
        self.dtype = dtype
        self.seed = seed

    @abstractmethod
    def sample(self):
        pass

    def __call__(self):
        return self.sample()
