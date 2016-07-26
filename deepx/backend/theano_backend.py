import os

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

import theano.tensor as T
import theano.sparse as sparse

from .backend_base import BackendBase

class TheanoBackend(BackendBase):

    # General purpose methods

    def _tensor(self, broadcastable, dtype=None, name=None):
        dtype = dtype or self.floatx()
        ttype = T.TensorType(dtype, broadcastable)
        device = self.get_current_device()
        if 'cpu' in device:
            device = 'cpu'
        return ttype(name).transfer(device)

    # def _variable(self, initial_value=None, trainable=True, name=None):
        # with self._device(self.get_current_device()):
            # return tf.Variable(initial_value=initial_value, trainable=trainable, name=name, target=self.get_current_device())

    # Tensorflow interface

    def placeholder(self, dtype, shape=None, name=None):
        if shape is None:
            raise Exception("Cannot specify None shape for Theano placeholder")
        broadcastable = []
        for s in shape:
            broadcastable.append(shape == 1)
        return self._tensor(broadcastable, dtype=dtype, name=name)

    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        if transpose_a:
            a = a.T
        if transpose_b:
            b = b.T
        if a_is_sparse or b_is_sparse:
            return sparse.dot(a, b)
        return T.dot(a, b)

    # Theano interface

    def scalar(self, name=None, dtype=None, shape=[]):
        return self._tensor([], dtype=dtype, name=name)

    def vector(self, name=None, dtype=None):
        return self._tensor([False], dtype=dtype, name=name)

    def matrix(self, name=None, dtype=None):
        return self._tensor([False, False], dtype=dtype, name=name)

    def tensor3(self, name=None, dtype=None):
        return self._tensor([False, False, False], dtype=dtype, name=name)

    def tensor4(self, name=None, dtype=None):
        return self._tensor([False, False, False, False], dtype=dtype, name=name)
