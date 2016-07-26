import tensorflow as tf
from .backend_base import BackendBase

class TensorflowBackend(BackendBase):

    # General purpose methods

    def _placeholder(self, dtype=None, shape=None, name=None):
        with self._device(self.get_current_device()):
            return tf.placeholder(dtype, shape=shape, name=name)

    def _variable(self, initial_value=None, trainable=True, name=None):
        with self._device(self.get_current_device()):
            return tf.Variable(initial_value=initial_value, trainable=trainable, name=name)

    def _device(self, name):
        return tf.device(name)

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
