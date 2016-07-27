import tensorflow as tf
from .backend_base import BackendBase, FunctionBase

class TensorflowFunction(FunctionBase):

    def __init__(self, *args, **kwargs):
        super(TensorflowFunction, self).__init__(*args, **kwargs)

    def __call__(self, *inputs):
        feed_dict = self.feed_dict(*inputs)
        return self.session.run(self.outputs, feed_dict=feed_dict)

class TensorflowBackend(BackendBase):

    def __init__(self, **kwargs):
        super(TensorflowBackend, self).__init__(**kwargs)
        self._session = self.session()

    # General purpose methods

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

    # Tensorflow interface

    def placeholder(self, dtype, shape=None, name=None):
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        return tf.matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=a_is_sparse, name=name)

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

    def function(self, inputs, outputs, updates=None):
        return TensorflowFunction(self._session, inputs, outputs)
