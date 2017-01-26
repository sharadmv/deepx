import logging
import six
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

def uses_device(method):
    method.uses_device = True
    return method

@six.add_metaclass(ABCMeta)
class FunctionBase(object):

    def __init__(self, backend, inputs, outputs, updates, lazy=True):
        self.backend = backend
        self.inputs = inputs
        self.outputs = outputs
        self.updates = updates
        self.lazy = lazy

    def get_session(self):
        return self.backend.get_current_session()

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

    @staticmethod
    @abstractmethod
    def use_device(method):
        pass

    @contextmanager
    def device(self, name):
        self.push_device(name)
        yield
        self.pop_device()

    def push_device(self, device):
        self._device_stack.append(device)
        logging.debug("Pushed device onto stack: {device}".format(device=device))

    def pop_device(self):
        device = self._device_stack.pop()
        logging.debug("Popped device from stack: {device}".format(device=device))

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

    @uses_device
    @abstractmethod
    def zeros(self, shape, dtype=None, name=None):
        pass

    @uses_device
    @abstractmethod
    def zeros_like(self, shape, dtype=None, name=None):
        pass

    @uses_device
    @abstractmethod
    def ones(self, shape, dtype=None, name=None):
        pass

    @uses_device
    @abstractmethod
    def ones_like(self, shape, dtype=None, name=None):
        pass

    @uses_device
    @abstractmethod
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        pass

    @uses_device
    @abstractmethod
    def random_uniform(self, shape, minval=0, maxval=None, dtype=None, seed=None):
        pass

    @uses_device
    @abstractmethod
    def tanh(self, x, name=None):
        pass

    @uses_device
    @abstractmethod
    def sigmoid(self, x, name=None):
        pass

    @uses_device
    @abstractmethod
    def relu(self, x, name=None):
        pass

    @uses_device
    @abstractmethod
    def softmax(self, x, T=1.0):
        pass

    @uses_device
    @abstractmethod
    def dropout(self, x, p, seed=None):
        pass

    @uses_device
    @abstractmethod
    def conv2d(self, x, kernel, strides=(1, 1), border_mode='same',
               image_shape=None, filter_shape=None):
        pass

    @uses_device
    @abstractmethod
    def pool2d(self, x, pool_size, strides=(1, 1),
               border_mode='valid', pool_mode='max'):
        pass

    @uses_device
    @abstractmethod
    def flatten(self, x):
        pass

    @uses_device
    @abstractmethod
    def sum(self, x, axis=None, keepdims=False):
        pass

    @uses_device
    @abstractmethod
    def mean(self, x, axis=None, keepdims=False):
        pass

    @uses_device
    @abstractmethod
    def log(self, x):
        pass

    @uses_device
    @abstractmethod
    def exp(self, x):
        pass

    @uses_device
    @abstractmethod
    def pow(self, x, a):
        pass

    @uses_device
    @abstractmethod
    def sqrt(x):
        pass

    @uses_device
    @abstractmethod
    def categorical_crossentropy(output, target, from_logits=False):
        pass

    @uses_device
    @abstractmethod
    def concatenate(self, tensors, axis=-1):
        pass

    @uses_device
    @abstractmethod
    def sort(self, tensor):
        pass

    @uses_device
    @abstractmethod
    def argmin(self, tensor, axis=0):
        pass

    # Tensorflow interface

    @abstractmethod
    def placeholder(self, dtype, shape=None, name=None):
        pass

    @abstractmethod
    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    @uses_device
    @abstractmethod
    def constant(self, value, dtype=None, shape=None):
        pass

    @abstractmethod
    def get_value(self, variable):
        pass

    @uses_device
    @abstractmethod
    def gather(self, params, indices):
        pass

    @uses_device
    @abstractmethod
    def gather_nd(self, params, indices):
        pass

    @uses_device
    @abstractmethod
    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        pass

    @uses_device
    @abstractmethod
    def expand_dims(self, x, dim=-1):
        pass

    @uses_device
    @abstractmethod
    def gradients(self, loss, variables):
        pass

    @uses_device
    @abstractmethod
    def square(self, x):
        pass

    @uses_device
    @abstractmethod
    def clip_by_value(self, x, low, high):
        pass

    @uses_device
    @abstractmethod
    def pack(self, values, axis=0, name='pack'):
        pass

    @uses_device
    @abstractmethod
    def reduce_max(self, x, axis=None, keepdims=False):
        pass

    # Theano interface

    @abstractmethod
    def dim(self, x):
        pass

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

    @uses_device
    @abstractmethod
    def dot(self, x, y):
        pass

    @uses_device
    @abstractmethod
    def sparse_dot(self, x, y):
        pass

    @abstractmethod
    def function(self, inputs, outputs, updates=None):
        pass

    @uses_device
    @abstractmethod
    def grad(self, loss, variables):
        pass

    @uses_device
    @abstractmethod
    def sqr(self, x):
        pass

    @uses_device
    @abstractmethod
    def max(self, x, axis=None, keepdims=False):
        pass

should_decorate = set()
for attr in dir(BackendBase):
    method = getattr(BackendBase, attr)
    if getattr(method, 'uses_device', False):
        should_decorate.add(attr)

class DeviceDecorator(ABCMeta):
    def __init__(cls, name, bases, dct):
        for method_name, method in dct.items():
            if method_name in should_decorate:
                setattr(cls, method_name, cls.use_device(method))
        type.__init__(cls, name, bases, dct)

