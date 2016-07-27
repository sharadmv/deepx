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
class FunctionBase(object):

    def __init__(self, session, inputs, outputs, lazy=True):
        self.session = session
        self.inputs = inputs
        self.outputs = outputs
        self.lazy = lazy

    def feed_dict(self, *inputs):
        return {i: input for i, input in zip(self.inputs, inputs)}

    @abstractmethod
    def __call__(self, *inputs):
        pass

@six.add_metaclass(ABCMeta)
class BackendBase(object):

    def __init__(self, use_cudnn=True):
        self._FLOATX = 'float32'
        self._EPSILON = 10e-8
        self._DEFAULT_DEVICE = '/gpu:0'
        self._device_stack = []
        self._initialized = False
        self.use_cudnn = use_cudnn

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

    @abstractmethod
    def session(self, allow_soft_placement=False, log_device_placement=False):
        pass


    def initialize(self):
        if not self._initialized:
            self._initialize()
            self._initialized = True

    @abstractmethod
    def _initialize(self):
        pass

    # Unified interface

    @abstractmethod
    def zeros(self, shape, dtype=None, name=None):
        pass

    @abstractmethod
    def zeros_like(self, shape, dtype=None, name=None):
        pass

    @abstractmethod
    def ones(self, shape, dtype=None, name=None):
        pass

    @abstractmethod
    def ones_like(self, shape, dtype=None, name=None):
        pass

    @abstractmethod
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        pass

    @abstractmethod
    def random_uniform(self, shape, minval=0, maxval=None, dtype=None, seed=None):
        pass

    @abstractmethod
    def tanh(self, x, name=None):
        pass

    @abstractmethod
    def sigmoid(self, x, name=None):
        pass

    @abstractmethod
    def relu(self, x, name=None):
        pass

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

    @abstractmethod
    def shared(self, value, name=None):
        pass

    @abstractmethod
    def dot(self, x, y):
        pass

    @abstractmethod
    def function(self, inputs, outputs, updates=None):
        pass
