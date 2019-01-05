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


class ExponentialFamily(Distribution):

    def __init__(self, parameters, parameter_type='regular'):
        self._parameter_cache = {}
        self._parameter_cache[parameter_type] = parameters
        super(ExponentialFamily, self).__init__()

    @abstractmethod
    def get_param_dim(self):
        pass

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        raise NotImplementedError

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        raise NotImplementedError

    @classmethod
    def internal_to_natural(cls, internal_parameters):
        raise NotImplementedError

    @classmethod
    def internal_to_regular(cls, internal_parameters):
        raise NotImplementedError

    @abstractmethod
    def log_z(self):
        pass

    @abstractmethod
    def log_h(self, x):
        pass

    @abstractmethod
    def sufficient_statistics(self, x):
        pass

    @abstractmethod
    def expected_sufficient_statistics(self):
        pass

    def get_parameters(self, parameter_type):
        if parameter_type not in self._parameter_cache:
            if parameter_type == 'regular':
                if 'natural' in self._parameter_cache:
                    self._parameter_cache['regular'] = self.natural_to_regular(self.get_parameters('natural'))
                elif 'internal' in self._parameter_cache:
                    self._parameter_cache['regular'] = self.internal_to_regular(self.get_parameters('internal'))
            elif parameter_type == 'natural':
                if 'regular' in self._parameter_cache:
                    self._parameter_cache['natural'] = self.regular_to_natural(self.get_parameters('regular'))
                elif 'internal' in self._parameter_cache:
                    self._parameter_cache['natural'] = self.internal_to_natural(self.get_parameters('internal'))
        return self._parameter_cache[parameter_type]
