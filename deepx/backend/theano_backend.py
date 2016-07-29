import numpy as np
import six
import os
from functools import wraps

CONTEXT_MAP = {
    '/gpu:0': 'cuda0',
    '/gpu:1': 'cuda1',
    '/gpu:2': 'cuda2',
    '/gpu:3': 'cuda3',
}
FLAGS = "contexts={contexts}".format(contexts=';'.join(["%s->%s" % (a, b) for (a, b) in CONTEXT_MAP.items()]))

if 'THEANO_FLAGS' in os.environ:
    flags = os.environ["THEANO_FLAGS"]
    flags += ",{flags}".format(flags=FLAGS)
else:
    flags = FLAGS
os.environ["THEANO_FLAGS"] = flags

import theano
import theano.tensor as T
import theano.sparse as sparse
from theano.sandbox.cuda import dnn
from theano.tensor.signal import pool

from .backend_base import BackendBase, FunctionBase, DeviceDecorator

class Session(object):

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass

class TheanoFunction(FunctionBase):

    def __init__(self, *args, **kwargs):
        super(TheanoFunction, self).__init__(*args, **kwargs)
        self.func = None
        if self.lazy is not True:
            self.create_function()

    def create_function(self):
        self.func = theano.function(self.inputs, self.outputs,
                                    allow_input_downcast=True)

    def __call__(self, *inputs):
        if self.func is None:
            self.create_function()
        return self.func(*inputs)

@six.add_metaclass(DeviceDecorator)
class TheanoBackend(BackendBase):

    def __init__(self, **kwargs):
        super(TheanoBackend, self).__init__(**kwargs)
        self._session = Session()

    # General purpose methods

    @classmethod
    def use_device(cls, method):
        @wraps(method)
        def func(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            if hasattr(result, 'transfer'):
                try:
                    return result.transfer(self.get_current_device())
                except ValueError:
                    return result
            return result
        return func

    def get_current_device(self):
        device = super(TheanoBackend, self).get_current_device()
        if 'cpu' in device:
            device = 'cpu'
        return device

    def _tensor(self, broadcastable, dtype=None, name=None):
        dtype = dtype or self.floatx()
        ttype = T.TensorType(dtype, broadcastable)
        return ttype(name).transfer(self.get_current_device())

    def _shared(self, value, name=None):
        value = np.array(value)
        return theano.shared(value, name=name, target=self.get_current_device())

    def session(self, allow_soft_placement=True, log_device_placement=True):
        return Session()

    def _initialize(self):
        return

    # Unified interface

    def zeros(self, shape, dtype=None, name=None):
        dtype = dtype or self.floatx()
        return T.zeros(shape, dtype=dtype)

    def zeros_like(self, x, dtype=None, name=None):
        result = T.zeros_like(x)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def ones(self, shape, dtype=None, name=None):
        dtype = dtype or self.floatx()
        return T.ones(shape, dtype=dtype)

    def ones_like(self, x, dtype=None, name=None):
        result = T.ones_like(x)
        if dtype is not None:
            result = result.astype(dtype)
        return result

    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        np.random.seed(seed)
        dtype = dtype or self.floatx()
        return np.random.normal(size=shape, loc=mean, scale=stddev).astype(dtype)

    def random_uniform(self, shape, minval=1, maxval=None, dtype=None, seed=None):
        np.random.seed(seed)
        dtype = dtype or self.floatx()
        if maxval is None and dtype == self.floatx():
            maxval = 1
        return np.random.uniform(size=shape, low=minval, high=maxval).astype(dtype)

    def tanh(self, x, name=None):
        return T.tanh(x)

    def sigmoid(self, x, name=None):
        return T.nnet.sigmoid(x)

    def relu(self, x, name=None):
        return T.nnet.relu(x)

    def conv2d(self, x, kernel, strides=[1, 1], border_mode='same'):
        if self.use_cudnn:
            if border_mode == 'same':
                assert(strides == [1, 1])
                np_kernel = kernel.eval()
                pad_x = (np_kernel.shape[2] - strides[0]) // 2
                pad_y = (np_kernel.shape[3] - strides[1]) // 2
                conv_out = dnn.dnn_conv(img=x,
                                        kerns=kernel,
                                        border_mode=(pad_x, pad_y))
            else:
                conv_out = dnn.dnn_conv(img=x,
                                        kerns=kernel,
                                        border_mode=border_mode,
                                        subsample=strides)
        else:
            if border_mode == 'same':
                th_border_mode = 'full'
                assert(strides == [1, 1])
            elif border_mode == 'valid':
                th_border_mode = 'valid'
            else:
                raise Exception('Border mode not supported: ' + str(border_mode))

            conv_out = T.nnet.conv2d(x, kernel,
                                        border_mode=th_border_mode,
                                        subsample=strides,
                                        image_shape=None,
                                        filter_shape=None)
            if border_mode == 'same':
                shift_x = (kernel.shape[2] - 1) // 2
                shift_y = (kernel.shape[3] - 1) // 2
                conv_out = conv_out[:, :,
                                    shift_x:x.shape[2] + shift_x,
                                    shift_y:x.shape[3] + shift_y]
        return conv_out

    def pool2d(self, x, pool_size, strides=(1, 1), border_mode='valid',
               pool_mode='max'):
        if border_mode == 'same':
            # TODO: add implementation for border_mode="same"
            raise Exception('border_mode="same" not supported with Theano.')
        elif border_mode == 'valid':
            ignore_border = False
            padding = (0, 0)
        else:
            raise Exception('Invalid border mode: ' + str(border_mode))

        pool_out = pool.pool_2d(x, ds=pool_size,
                                ignore_border=ignore_border,
                                padding=padding,
                                mode=pool_mode
                                )
        return pool_out

    def flatten(self, x):
        return T.reshape(x, (x.shape[0], T.prod(x.shape) // x.shape[0]))

    def mean(self, x, axis=None, keepdims=False):
        return T.mean(x, axis=axis, keepdims=keepdims)

    def log(self, x):
        return T.log(x)

    # Tensorflow interface

    def placeholder(self, dtype, shape=None, name=None):
        if shape is None:
            raise Exception("Cannot specify None shape for Theano placeholder")
        broadcastable = []
        for s in shape:
            broadcastable.append(shape == 1)
        return self._tensor(broadcastable, dtype=dtype, name=name)

    def variable(self, initial_value=None, trainable=True, name=None):
        return self._shared(initial_value, name=name)

    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        if transpose_a:
            a = a.T
        if transpose_b:
            b = b.T
        if a_is_sparse or b_is_sparse:
            return sparse.dot(a, b)
        return T.dot(a, b)

    def expand_dims(self, x, dim=-1):
        '''Add a 1-sized dimension at index "dim".
        '''
        pattern = [i for i in range(x.type.ndim)]
        if dim < 0:
            if x.type.ndim == 0:
                dim = 0
            else:
                dim = dim % x.type.ndim + 1
        pattern.insert(dim, 'x')
        return x.dimshuffle(pattern)

    # Theano interface

    def scalar(self, name=None, dtype=None, shape=[]):
        return self._tensor([], dtype=dtype, name=name)

    def vector(self, name=None, dtype=None, shape=[]):
        return self._tensor([False], dtype=dtype, name=name)

    def matrix(self, name=None, dtype=None, shape=[]):
        return self._tensor([False, False], dtype=dtype, name=name)

    def tensor3(self, name=None, dtype=None, shape=[]):
        return self._tensor([False, False, False], dtype=dtype, name=name)

    def tensor4(self, name=None, dtype=None, shape=[]):
        return self._tensor([False, False, False, False], dtype=dtype, name=name)

    def shared(self, value, name=None):
        return self._shared(value, name=name)

    def dot(self, x, y):
        return T.dot(x, y)

    def function(self, inputs, outputs, updates=None):
        return TheanoFunction(self._session, inputs, outputs)
