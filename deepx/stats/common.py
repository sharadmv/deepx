import six
from abc import ABCMeta, abstractmethod

@six.add_metaclass(ABCMeta)
class Distribution(object):

    @abstractmethod
    def sample(self, num_samples=1):
        pass

    @abstractmethod
    def expected_value(self):
        pass


    @abstractmethod
    def log_likelihood(self, x):
        pass
