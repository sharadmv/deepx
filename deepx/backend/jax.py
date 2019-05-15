import copy
import jax.random as random
import logging
import jax.numpy as np
from jax import lax
import six
from functools import wraps
from contextlib import contextmanager

from .backend_base import BackendBase, FunctionBase, DeviceDecorator

class PRNG(object):

    def __init__(self, seed):
        key = random.PRNGKey(0)
        self.key = key
        self.subkey = key

    def __next__(self):
        self.key, subkey = random.split(self.key)
        return subkey

@six.add_metaclass(DeviceDecorator)
class JaxBackend(BackendBase):

    def __init__(self, **kwargs):
        super(JaxBackend, self).__init__(**kwargs)
        self.core = np
        self._sessions = []
        self.rng = PRNG(0)

    # General purpose methods

    @classmethod
    def use_device(cls, method):
        @wraps(method)
        def func(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            return result
        return func

    def cpu(self, id=0):
        return 'cpu:%u' % id

    def gpu(self, id=0):
        return 'cuda:%u' % id

    @property
    def int32(self):
        return np.int32

    @property
    def float32(self):
        return np.float32

    def _placeholder(self, dtype=None, shape=None, name=None):
        raise NotImplementedError

    def _variable(self, initial_value=None, trainable=True, name=None):
        return initial_value

    def _device(self, name):
        raise NotImplementedError

    def create_session(self, **kwargs):
        raise NotImplementedError

    @contextmanager
    def session(self, **kwargs):
        raise NotImplementedError

    def interactive_session(self, graph=None, **kwargs):
        raise NotImplementedError

    def get_current_session(self):
        raise NotImplementedError

    def _initialize(self, sess):
        raise Exception('No current session')

    # Unified interface

    def coerce(self, x, dtype=None):
        return np.array(x).astype(dtype)

    def cast(self, x, dtype):
        return x.astype(dtype)

    def dtype(self, x):
        return x.dtype

    def shape(self, x):
        return x.shape

    def rank(self, x):
        return len(x.shape)

    def abs(self, x):
        return np.abs(x)

    def set_value(self, x, value):
        raise NotImplementedError

    def floatx(self, as_string=False):
        if as_string:
            return super(JaxBackend, self).floatx()
        return getattr(np, super(JaxBackend, self).floatx())

    def zeros(self, shape, dtype=None, name=None):
        dtype = dtype or self.floatx()
        if not isinstance(shape, int):
            shape = tuple(shape)
        return np.zeros(shape, dtype=dtype)

    def zeros_like(self, x, dtype=None, name=None):
        return np.zeros_like(x, dtype=dtype)

    def ones(self, shape, dtype=None, name=None):
        dtype = dtype or self.floatx()
        return np.ones(shape, dtype=dtype)

    def ones_like(self, x, dtype=None, name=None):
        return np.ones_like(x, dtype=dtype)

    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        dtype = dtype or self.floatx()
        shape = list(shape)
        seed = next(self.rng)
        return mean + stddev * random.normal(seed, shape, dtype=dtype)

    def random_truncated_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        raise NotImplementedError

    def random_uniform(self, shape, minval=1.0, maxval=None, dtype=None, seed=None):
        dtype = dtype or self.floatx()
        if maxval is None:
            minval, maxval = 0.0, minval
        shape = list(shape)
        seed = next(self.rng)
        samples = random.uniform(seed, shape, dtype=dtype)
        return samples * (maxval - minval) + minval

    def random_binomial(self, shape, p=0.5, dtype=None):
        raise NotImplementedError

    def random_gamma(self, shape, alpha, beta=None):
        raise NotImplementedError

    def tanh(self, x, name=None):
        return np.tanh(x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def relu(self, x, alpha=0.):
        return np.maximum(x, 0.)

    def softmax(self, x, T=1.0):
        unnormalized = np.exp(x - x.max(-1, keepdims=True))
        return unnormalized / unnormalized.sum(-1, keepdims=True)

    def softplus(self, x):
        return np.logaddexp(x, 0.)

    def dropout(self, x, p, seed=None):
        seed = next(self.rng)
        p = 1 - p
        keep = random.bernoulli(seed, p, x.shape)
        return np.where(keep, x / p, 0)

    def conv2d(self, x, kernel, strides=(1, 1), border_mode='same',
               image_shape=None, filter_shape=None):
        return lax.conv_general_dilated(x, kernel, strides, border_mode,
                                    dimension_numbers=("NHWC", "HWIO", "NHWC"))

    def conv2d_transpose(self, x, kernel, dim_out, strides=(1, 1), border_mode='same'):
        raise NotImplementedError

    def pool2d(self, x, pool_size, strides=(1, 1),
               border_mode='valid', pool_mode='max'):
        dims = (1,) + pool_size + (1,)
        strides = (1,) + strides + (1,)
        return lax.reduce_window(x, -np.inf, lax.max, dims, strides, border_mode)

    def flatten(self, x, leading=1):
        raise NotImplementedError

    def split(self, x, num_splits, axis=None):
        raise NotImplementedError

    def reshape(self, x, shape):
        if not isinstance(x, int):
            shape = tuple(shape)
        shape = tuple(-1 if s is None else s for s in shape)
        return np.reshape(x, tuple(map(int, shape)))

    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, dim=axis, keepdim=keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return np.prod(x, dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return np.mean(x, axis=axis, keepdims=keepdims)

    def batch_norm(self, x, beta, gamma):
        raise NotImplementedError

    def log(self, x):
        return np.log(x)

    def log1p(self, x):
        return np.log1p(x)

    def exp(self, x):
        return np.exp(x)

    def pow(self, x, a):
        return np.pow(x, a)

    def mul(self, x, y):
        return np.mul(x, y)

    def sqrt(self, x):
        return np.sqrt(x)

    def categorical_crossentropy(self, output, target, from_logits=False):
        if from_logits:
            raise NotImplementedError
        return -np.mean(np.log(output) * target, axis=-1)

    def binary_crossentropy(self, output, target, from_logits=False):
        raise NotImplementedError

    def concatenate(self, tensors, axis=-1):
        values = [self.coerce(v, dtype=self.floatx()) for v in tensors]
        return np.concatenate(values, axis=int(axis))

    def sort(self, tensor, axis=-1):
        return np.sort(tensor, axis=axis)

    def argmin(self, tensor, axis=0):
        return np.argmin(tensor, axis=axis)

    def map(self, function, input):
        return map(function, input)

    def rnn(self, step_function, input, initial_states, **kwargs):
        input = np.swapaxes(input, 0, 1)
        def step(state, input_):
            output, state = step_function(input_, state, **kwargs)
            return state, output
        state, output = self.scan(step, input, initial_states)
        return np.swapaxes(output, 0, 1)

    def while_loop(self, condition, body, loop_vars, **kwargs):
        raise NotImplementedError

    def scan(self, fn, elems, initializer=None):
        return lax.scan(fn, initializer, elems)

    def logdet(self, A, **kwargs):
        A = (A + self.matrix_transpose(A)) / 2.
        term = np.log(np.diag(self.cholesky(A, **kwargs)))
        return 2 * np.sum(term, axis=-1)

    def einsum(self, subscripts, *operands):
        return np.einsum(subscripts, operands)

    def cholesky(self, A, lower=True, warn=False, correct=True):
        return np.linalg.cholesky(A)

    # Tensorflow interface

    def placeholder(self, dtype, shape=None, name=None):
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    def assign(self, a, b):
        raise NotImplementedError

    def to_float(self, x):
        return np.array(x, dtype=self.floatx())

    def constant(self, value, dtype=None, shape=None):
        return np.array(value).astype(dtype)

    def get_shape(self, x):
        return list(x.shape)

    def get_value(self, variable):
        return variable.numpy()

    def concat(self, values, axis=-1):
        return self.concatenate(values, axis=axis)

    def gather(self, params, indices):
        return params[indices]

    def gather_nd(self, params, indices):
        raise NotImplementedError

    def equal(self, x, y):
        return np.equal(x, y)

    def logical_and(self, x, y):
        return x and y

    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        if transpose_a:
            a = self.matrix_transpose(a)
        if transpose_b:
            b = self.matrix_transpose(b)
        return np.matmul(a, b)

    def trace(self, a):
        return np.trace(a)

    def transpose(self, a, perm=None):
        return a.permute(*perm)

    def matrix_transpose(self, a):
        return np.swapaxes(a, -1, -2)

    def matrix_diag(self, a):
        raise NotImplementedError

    def matrix_diag_part(self, a):
        raise NotImplementedError

    def set_diag(self, input, diagonal):
        raise NotImplementedError

    def band_part(self, input, num_lower, num_upper):
        raise NotImplementedError


    def vec(self, A):
        A = self.matrix_transpose(A)
        leading_dim = self.shape(A)[:-2]
        return self.reshape(A, self.concat([
            leading_dim,
            [-1]
        ], 0))

    def unvec(self, v, m, n):
        leading_dim = self.shape(v)[:-1]
        return self.matrix_transpose(self.reshape(v, self.concat([
            leading_dim,
            [n, m]
        ], 0)))

    def kronecker(self, A, B):
        C = (A[..., None, None] * B[..., None, None, :, :])
        blocks = [
            self.unstack(a, axis=-3 % len(a.shape)) for a in
            self.unstack(C, axis=-4 % len(C.shape))
        ]
        return self.concat([
            self.concat(a, -1) for a in blocks
        ], -2)

    def block_sum(self, X, m, n):
        leading_dim = self.shape(X)[:-2]
        block_sum = self.zeros(self.concat([leading_dim, [m, m]], 0))
        for i in range(n):
            block_sum += X[..., i*m:(i+1)*m, i*m:(i+1)*m]
        return block_sum

    def block_trace(self, X, m, n):
        blocks = []
        for i in range(n):
            blocks.append([])
            for j in range(n):
                block = self.trace(X[..., i*m:(i+1)*m, j*m:(j+1)*m])
                blocks[-1].append(block)
        return self.pack([
            self.pack([
                b for b in block
            ])
            for block in blocks
        ])

    def kronecker_vec(self, X, m, n):
        leading_dim = self.shape(X)[:-2]
        blocks = []
        for i in range(n):
            blocks.append([])
            for j in range(m):
                idx = i * m + j
                block = self.matrix_transpose(self.reshape(X[..., idx, :], self.concat([leading_dim, [n, m]], 0)))
                blocks[-1].append(block)
        return self.concat([self.concat(b, -2) for b in blocks], -1)

    def lower_triangular(self, a):
        raise NotImplementedError

    def matrix_inverse(self, a):
        return np.linalg.inv(a)

    def expand_dims(self, x, dim=-1):
        return np.expand_dims(x, dim=dim)

    def tile(self, input, multiples):
        return np.tile(input, multiples)

    def gradients(self, loss, variables):
        raise NotImplementedError("Please use `jax.grad`")

    def square(self, x):
        return np.pow(x, 2)

    def clip_by_value(self, x, low, high):
        return x.clamp(low, high)

    def stack(self, values, axis=0, name='stack'):
        return np.stack(values, dim=axis)

    def unstack(self, values, num=None, axis=0):
        return np.unstack(values, dim=axis)

    def pack(self, *args, **kwargs):
        return self.stack(*args, **kwargs)

    def unpack(self, *args, **kwargs):
        return self.unstack(*args, **kwargs)

    def reduce_max(self, x, axis=None, keepdims=False):
        return np.max(x, axis=axis, keepdim=keepdims)

    def reduce_logsumexp(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def matrix_solve(self, matrix, rhs, adjoint=None):
        return np.linalg.solve(matrix, rhs)

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

    def arange(self, start, stop=None, step=None):
        return self.range(start, stop=stop, step=step)

    def sparse_dot(self, x, y):
        raise NotImplementedError

    def dot(self, x, y):
        return np.dot(x, y)

    def outer(self, x, y):
        if len(self.get_shape(x)) == 0:
            return x * y
        return x[...,:,None] * y[...,None,:]

    def eye(self, d, batch_shape=None):
        out = np.eye(d)
        if batch_shape is not None:
            for _ in batch_shape:
                out = out[None]
            return self.tile(out, batch_shape + [1, 1])
        return out

    def function(self, inputs, outputs, updates=[]):
        raise NotImplementedError

    def grad(self, loss, variables):
        return self.gradients(loss, variables)

    def sqr(self, x):
        return self.square(x)

    def argmax(self, x, axis=None):
        return np.argmax(x, axis=axis)

    def max(self, x, axis=None, keepdims=False):
        return self.reduce_max(x, axis=axis, keepdims=keepdims)

    def logsumexp(self, x, axis=None, keepdims=False):
        return self.reduce_logsumexp(x, axis=axis, keepdims=keepdims)

    def switch(self, condition, then_expression, else_expression):
        raise NotImplementedError

    def alloc(self, value, shape, unbroadcast=None, dtype=None):
        raise NotImplementedError

    def range(self, start, limit=None, delta=1):
        return np.arange(start, limit, step=delta)

    def solve(self, a, b):
        return self.matrix_solve(a, b)

    def one_hot(self, indices, depth):
        raise NotImplementedError

    # Science methods

    def gammaln(self, x):
        raise NotImplementedError

    def multigammaln(self, a, p):
        p = self.to_float(p)
        p_ = self.cast(p, 'int32')
        a = a[..., None]
        i = self.to_float(self.range(1, p_ + 1))
        term1 = p * (p - 1) / 4. * self.log(np.pi)
        term2 = self.gammaln(a - (i - 1) / 2.)
        return term1 + self.sum(term2, axis=-1)

    def digamma(self, a):
        raise NotImplementedError
