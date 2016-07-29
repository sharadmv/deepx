import numpy as np
import six
import tensorflow as tf
from functools import wraps

from .backend_base import BackendBase, FunctionBase, DeviceDecorator


class TensorflowFunction(FunctionBase):

    def __init__(self, *args, **kwargs):
        super(TensorflowFunction, self).__init__(*args, **kwargs)
        with tf.control_dependencies(self.outputs):
            self.updates = [tf.assign(k, v) for k, v in self.updates]

    def __call__(self, *inputs):
        feed_dict = self.feed_dict(*inputs)
        result = self.session.run(self.outputs + self.updates, feed_dict=feed_dict)
        if len(self.outputs) == 1:
            return result[0]
        return result[:len(self.outputs)]

@six.add_metaclass(DeviceDecorator)
class TensorflowBackend(BackendBase):

    def __init__(self, **kwargs):
        super(TensorflowBackend, self).__init__(**kwargs)
        self._session = self.session()

    # General purpose methods

    @classmethod
    def use_device(cls, method):
        @wraps(method)
        def func(self, *args, **kwargs):
            with tf.device(self.get_current_device()):
                result = method(self, *args, **kwargs)
            return result
        return func

    def _placeholder(self, dtype=None, shape=None, name=None):
        with self._device(self.get_current_device()):
            return tf.placeholder(dtype, shape=shape, name=name)

    def _variable(self, initial_value=None, trainable=True, name=None):
        with self._device(self.get_current_device()):
            return tf.Variable(initial_value=initial_value, trainable=trainable, name=name)

    def _device(self, name):
        return tf.device(name)

    def session(self, allow_soft_placement=True, log_device_placement=True):
        config_proto = tf.ConfigProto(allow_soft_placement=True,
                                      log_device_placement=True)
        return tf.Session(config=config_proto)


    def _initialize(self):
        self._session.run(tf.initialize_all_variables())

    # Unified interface

    def zeros(self, shape, dtype=None, name=None):
        dtype = dtype or self.floatx()
        return tf.zeros(shape, dtype=dtype, name=name)

    def zeros_like(self, x, dtype=None, name=None):
        return tf.zeros_like(x, dtype=dtype, name=name)

    def ones(self, shape, dtype=None, name=None):
        dtype = dtype or self.floatx()
        return tf.ones(shape, dtype=dtype, name=name)

    def ones_like(self, x, dtype=None, name=None):
        return tf.ones_like(x, dtype=dtype, name=name)

    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        dtype = dtype or self.floatx()
        return tf.random_normal(shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)

    def random_uniform(self, shape, minval=0, maxval=None, dtype=None, seed=None):
        dtype = dtype or self.floatx()
        return tf.random_uniform(shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)

    def tanh(self, x, name=None):
        return tf.tanh(x, name=name)

    def sigmoid(self, x, name=None):
        return tf.sigmoid(x, name=name)

    def relu(self, x, name=None):
        return tf.nn.relu(x, name=name)

    def softmax(self, x, T=1.0):
        return tf.nn.softmax(x)

    def conv2d(self, x, kernel, strides=(1, 1), border_mode='same',
               image_shape=None, filter_shape=None):
        '''
        Run on cuDNN if available.
        border_mode: string, "same" or "valid".
        dim_ordering: whether to use Theano or TensorFlow dimension ordering
        in inputs/kernels/ouputs.
        '''
        if border_mode == 'same':
            padding = 'SAME'
        elif border_mode == 'valid':
            padding = 'VALID'
        else:
            raise Exception('Invalid border mode: ' + str(border_mode))

        strides = (1,) + strides + (1,)

        if self.floatx() == 'float64':
            x = tf.cast(x, 'float32')
            kernel = tf.cast(kernel, 'float32')

        x = tf.transpose(x, (0, 2, 3, 1))
        kernel = tf.transpose(kernel, (2, 3, 1, 0))
        x = tf.nn.conv2d(x, kernel, strides, padding=padding)
        x = tf.transpose(x, (0, 3, 1, 2))

        if self.floatx() == 'float64':
            x = tf.cast(x, 'float64')
        return x

    def pool2d(self, x, pool_size, strides=(1, 1),
               border_mode='valid', pool_mode='max'):
        '''
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        border_mode: one of "valid", "same".
        dim_ordering: one of "th", "tf".
        '''
        if border_mode == 'same':
            padding = 'SAME'
        elif border_mode == 'valid':
            padding = 'VALID'
        else:
            raise Exception('Invalid border mode: ' + str(border_mode))

        strides = (1,) + strides + (1,)
        pool_size = (1,) + pool_size + (1,)

        if self.floatx() == 'float64':
            x = tf.cast(x, 'float32')

        x = tf.transpose(x, (0, 2, 3, 1))
        if pool_mode == 'max':
            x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
        elif pool_mode == 'avg':
            x = tf.nn.avg_pool(x, pool_size, strides, padding=padding)
        else:
            raise Exception('Invalid pooling mode: ' + str(pool_mode))
        x = tf.transpose(x, (0, 3, 1, 2))

        if self.floatx() == 'float64':
            x = tf.cast(x, 'float64')
        return x

    def flatten(self, x):
        return tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])

    def mean(self, x, axis=None, keepdims=False):
        if axis is not None and axis < 0:
            axis = axis % len(x.get_shape())
        if x.dtype.base_dtype == tf.bool:
            x = tf.cast(x, self.floatx())
        return tf.reduce_mean(x, reduction_indices=axis, keep_dims=keepdims)

    def log(self, x):
        return tf.log(x)

    def exp(self, x):
        return tf.exp(x)

    def sqrt(self, x):
        x = tf.clip_by_value(x,
                             tf.cast(0., dtype=self.floatx()),
                             tf.cast(np.inf, dtype=self.floatx()))
        return tf.sqrt(x)

    def categorical_crossentropy(self, output, target, from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    reduction_indices=len(output.get_shape())-1,
                                    keep_dims=True)
            output = tf.clip_by_value(output, tf.cast(self.epsilon(), dtype=self.floatx()),
                                    tf.cast(1.- self.epsilon(), dtype=self.floatx()))
            return - tf.reduce_sum(target * tf.log(output),
                                   reduction_indices=len(output.get_shape())-1)
        else:
            return tf.nn.softmax_cross_entropy_with_logits(output, target)

    # Tensorflow interface

    def placeholder(self, dtype, shape=None, name=None):
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=a_is_sparse, name=name)

    def expand_dims(self, x, dim=-1):
        return tf.expand_dims(x, dim)

    def gradients(self, loss, variables):
        return tf.gradients(loss, variables)

    def square(self, x):
        return tf.square(x)

    # Theano interface

    def scalar(self, name=None, dtype=None, shape=[]):
        dtype = dtype or self.floatx()
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def vector(self, name=None, dtype=None, shape=[None]):
        dtype = dtype or self.floatx()
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def matrix(self, name=None, dtype=None, shape=[None, None]):
        dtype = dtype or self.floatx()
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def tensor3(self, name=None, dtype=None, shape=[None, None, None]):
        dtype = dtype or self.floatx()
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def tensor4(self, name=None, dtype=None, shape=[None, None, None, None]):
        dtype = dtype or self.floatx()
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def shared(self, value, name=None):
        return self._variable(initial_value=value, name=name)

    def dot(self, x, y):
        return tf.matmul(x, y)

    def function(self, inputs, outputs, updates=[]):
        return TensorflowFunction(self._session, inputs, outputs, updates)
    def grad(self, loss, variables):
        return tf.gradients(loss, variables)

    def sqr(self, x):
        return tf.square(x)


