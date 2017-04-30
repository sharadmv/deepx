import copy
import numpy as np
import six
import tensorflow as tf
from functools import wraps
from contextlib import contextmanager

from .backend_base import BackendBase, FunctionBase, DeviceDecorator

class TensorflowFunction(FunctionBase):

    def __init__(self, *args, **kwargs):
        super(TensorflowFunction, self).__init__(*args, **kwargs)
        with tf.control_dependencies(self.outputs):
            self.updates = [tf.assign(k, v) for k, v in self.updates]

    def __call__(self, *inputs):
        feed_dict = self.feed_dict(*inputs)
        result = self.session.get_current_session().run(self.outputs + self.updates, feed_dict=feed_dict)
        if len(self.outputs) == 1:
            return result[0]
        return result[:len(self.outputs)]

@six.add_metaclass(DeviceDecorator)
class TensorflowBackend(BackendBase):

    def __init__(self, **kwargs):
        super(TensorflowBackend, self).__init__(**kwargs)
        self.core = tf
        self._sessions = []

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

    def create_session(self, **kwargs):
        config_proto = tf.ConfigProto(**kwargs)
        sess = tf.Session(config=config_proto)
        self._initialize(sess)
        return sess

    @contextmanager
    def session(self, **kwargs):
        config_proto = tf.ConfigProto(**kwargs)
        with tf.Session(config=config_proto) as sess:
            self._sessions.append(sess)
            self._initialize(sess)
            yield sess
            self._sessions.pop()

    def get_current_session(self):
        if len(self._sessions) == 0:
            raise Exception('No current session')
        return self._sessions[-1]

    def _initialize(self, sess):
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

    # Unified interface

    def shape(self, x):
        return tf.shape(x)

    def set_value(self, x, value):
        tf.assign(x, np.asarray(value)).op.run(session=self.get_current_session())

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

    def random_binomial(self, shape, p=0.5, dtype=None):
        dtype = dtype or self.floatx()
        return tf.where(tf.random_uniform(shape, dtype=dtype) <= p,
                                           tf.ones(shape, dtype=dtype),
                                           tf.zeros(shape, dtype=dtype))

    def tanh(self, x, name=None):
        return tf.tanh(x, name=name)

    def sigmoid(self, x, name=None):
        return tf.sigmoid(x, name=name)

    def relu(self, x, name=None):
        return tf.nn.relu(x, name=name)

    def softmax(self, x, T=1.0):
        return tf.nn.softmax(x)

    def dropout(self, x, p, seed=None):
        retain_prob = 1. - p
        if seed is None:
            seed = np.random.randint(10e6)
        return tf.nn.dropout(x * 1., retain_prob, seed=seed)

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

        x = tf.nn.conv2d(x, kernel, strides, padding=padding)

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

        if pool_mode == 'max':
            x = tf.nn.max_pool(x, pool_size, strides, padding=padding)
        elif pool_mode == 'avg':
            x = tf.nn.avg_pool(x, pool_size, strides, padding=padding)
        else:
            raise Exception('Invalid pooling mode: ' + str(pool_mode))

        if self.floatx() == 'float64':
            x = tf.cast(x, 'float64')
        return x

    def flatten(self, x, leading=1):
        leading_dim = self.shape(x)[:leading]
        new_shape = tf.concat([leading_dim, [-1]], 0)
        return tf.reshape(x, new_shape)

    def split(self, x, num_splits, axis=None):
        axis = axis % len(x.get_shape())
        return tf.split(x, num_splits, axis=axis)

    def reshape(self, x, shape):
        return tf.reshape(x, shape)

    def sum(self, x, axis=None, keepdims=False):
        if axis is not None and axis < 0:
            axis = axis % len(x.get_shape())
        if x.dtype.base_dtype == tf.bool:
            x = tf.cast(x, self.floatx())
        return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)


    def mean(self, x, axis=None, keepdims=False):
        if axis is not None and axis < 0:
            axis = axis % len(x.get_shape())
        if x.dtype.base_dtype == tf.bool:
            x = tf.cast(x, self.floatx())
        return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)

    def batch_norm(self, x, beta, gamma):
        mean, variance = tf.nn.moments(x, [0])
        normed = tf.nn.batch_normalization(tf.identity(x), mean, variance, beta, gamma, self.epsilon())
        return normed

    def log(self, x):
        return tf.log(x)

    def exp(self, x):
        return tf.exp(x)

    def pow(self, x, a):
        return tf.pow(x, a)

    def sqrt(self, x):
        x = tf.clip_by_value(x,
                             tf.cast(0., dtype=self.floatx()),
                             tf.cast(np.inf, dtype=self.floatx()))
        return tf.sqrt(x)

    def categorical_crossentropy(self, output, target, from_logits=False):
        if not from_logits:
            output /= tf.reduce_sum(output,
                                    axis=len(output.get_shape())-1,
                                    keep_dims=True)
            output = tf.clip_by_value(output, tf.cast(self.epsilon(), dtype=self.floatx()),
                                    tf.cast(1.- self.epsilon(), dtype=self.floatx()))
            return - tf.reduce_sum(target * tf.log(output),
                                   axis=len(output.get_shape()) - 1)
        else:
            return tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target)

    def concatenate(self, tensors, axis=-1, concat=False):
        if concat:
            return tf.stack(tensors)
        if axis < 0:
            axis = axis % len(tensors[0].get_shape())
        return tf.concat(axis=axis, values=tensors)

    def sort(self, tensor):
        values, indices = tf.nn.top_k(-tensor, k=tf.shape(tensor)[0])
        return -values, indices

    def argmin(self, tensor, axis=0):
        return tf.argmin(tensor, axis=axis)

    def map(self, function, input):
        return tf.map_fn(function, input)

    def rnn(self, step_function, input, initial_states):
        def step(accumulator, value):
            _, new_accumulator = step_function(value, accumulator)
            return new_accumulator
        result = tf.scan(step, input, initial_states)
        return result

    # Tensorflow interface

    def placeholder(self, dtype, shape=None, name=None):
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    def constant(self, value, dtype=None, shape=None):
        return tf.constant(value, dtype=dtype, shape=shape)

    def get_shape(self, x):
        return [a.value for a in x.get_shape()]

    def get_value(self, variable):
        return self.get_current_session().run(variable)

    def gather(self, params, indices):
        return tf.gather(params, indices)

    def gather_nd(self, params, indices):
        return tf.gather_nd(params, indices)

    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=a_is_sparse, name=name)

    def expand_dims(self, x, dim=-1):
        return tf.expand_dims(x, dim)

    def gradients(self, loss, variables):
        return tf.gradients(loss, variables)

    def square(self, x):
        return tf.square(x)

    def clip_by_value(self, x, low, high):
        return tf.clip_by_value(x, low, high)

    def pack(self, values, axis=0, name='pack'):
        return tf.stack(values, axis=axis, name=name)

    def reduce_max(self, x, axis=None, keep_dims=False):
        return tf.reduce_max(x, axis=axis, keep_dims=keep_dims)

    # Theano interface

    def dim(self, x):
        return len(x.get_shape())

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

    def sparse_dot(self, x, y):
        return tf.sparse_tensor_dense_matmul(x, y)

    def dot(self, x, y):
        return tf.matmul(x, y)

    def outer(self, x, y):
        return x[...,:,None] * y[...,None,:]
        if len(x.get_shape()) == 1:
            x = tf.expand_dims(x, 1)
        if len(y.get_shape()) == 1:
            y = tf.expand_dims(y, 0)
        return tf.matmul(x, y)

    def eye(self, d):
        if not (isinstance(d, list) or isinstance(d, tuple)):
            d = [d]
        return tf.diag(tf.ones(d))

    def function(self, inputs, outputs, updates=[]):
        return TensorflowFunction(self, inputs, outputs, updates)

    def grad(self, loss, variables):
        return tf.gradients(loss, variables)

    def sqr(self, x):
        return tf.square(x)

    def max(self, x, axis=None, keepdims=False):
        return tf.reduce_max(x, axis=axis, keep_dims=keepdims)

    def switch(self, condition, then_expression, else_expression):
        '''Switches between two operations depending on a scalar value (int or bool).
        Note that both `then_expression` and `else_expression`
        should be symbolic tensors of the *same shape*.
        # Arguments
            condition: scalar tensor.
            then_expression: TensorFlow operation.
            else_expression: TensorFlow operation.
        '''
        return tf.where(condition, then_expression, else_expression)

    def alloc(self, value, shape, unbroadcast=None, dtype=None):
        dtype = dtype or self.floatx()
        vals = tf.fill(tf.stack(shape), np.array(value).astype(dtype))
        new_shape = []
        for s in shape:
            if isinstance(s, tf.Tensor):
                new_shape.append(None)
            else:
                new_shape.append(s)
        vals.set_shape(new_shape)
        return vals
