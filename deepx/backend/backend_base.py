import logging
import six
from abc import ABCMeta, abstractmethod

class Device(object):

    def __init__(self, name, backend):
        self.name = name
        self.backend = backend

    def __enter__(self):
        logging.debug("Pushing device onto stack: {device}".format(device=self.name))
        self.backend.push_device(self.name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.debug("Popping device from stack: {device}".format(device=self.name))
        self.backend.pop_device()

@six.add_metaclass(ABCMeta)
class BackendBase(object):

    def __init__(self):
        self._FLOATX = 'float32'
        self._EPSILON = 10e-8
        self._DEFAULT_DEVICE = '/gpu:0'
        self._device_stack = []

    # Global functions

    def epsilon(self):
        return self._EPSILON

    def set_epsilon(self, e):
        self._EPSILON = e

    def floatx(self):
        return self._FLOATX

    def set_floatx(self, floatx):
        if floatx not in {'float32', 'float64'}:
            raise Exception('Unknown floatx type: ' + str(floatx))
        self._FLOATX = str(floatx)

    def get_default_device(self):
        return self._DEFAULT_DEVICE

    def set_default_device(self, default_device):
        self._DEFAULT_DEVICE = default_device

    # General methods

    def device(self, name):
        return Device(name, self)

    def push_device(self, device):
        self._device_stack.append(device)

    def pop_device(self):
        self._device_stack.pop()

    def get_current_device(self):
        if len(self._device_stack) == 0:
            return self.get_default_device()
        return self._device_stack[-1]

    # Tensorflow interface

    @abstractmethod
    def placeholder(self, dtype, shape=None, name=None):
        pass

    @abstractmethod
    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    @abstractmethod
    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        pass

    # Theano interface

    @abstractmethod
    def scalar(self, name=None, dtype=None):
        pass

    @abstractmethod
    def vector(self, name=None, dtype=None):
        pass

    @abstractmethod
    def matrix(self, name=None, dtype=None):
        pass

    @abstractmethod
    def tensor3(self, name=None, dtype=None):
        pass

    @abstractmethod
    def tensor4(self, name=None, dtype=None):
        pass
