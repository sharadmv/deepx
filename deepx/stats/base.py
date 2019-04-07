import six
from abc import ABCMeta, abstractmethod

@six.add_metaclass(ABCMeta)
class Distribution(object):

    @abstractmethod
    def sample(self, num_samples=[]):
        pass

    @abstractmethod
    def log_likelihood(self, x):
        pass

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def entropy(self):
        pass

    def expected_value(self):
        return self.mean()
