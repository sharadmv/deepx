import copy
import torch
import logging
import numpy as np
import six
from functools import wraps
from contextlib import contextmanager

from .backend_base import BackendBase, FunctionBase, DeviceDecorator

class TorchPlaceholder(object):
    def __init__(self, dtype, shape, device, name=None):
        self.dtype = dtype
        self.shape = shape
        self.device = device
        self.name = name

class TorchSession(object):
    pass

class PyTorchBackend(BackendBase):

    def __init__(self, **kwargs):
        super(PyTorchBackend, self).__init__(**kwargs)
        self.core = torch
        self._sessions = []

    # General purpose methods

    @classmethod
    def use_device(cls, method):
        @wraps(method)
        def func(self, *args, **kwargs):
            return result.to(self.get_current_device())
        return func

    def cpu(self, id=0):
        return 'cpu:%u' % id

    def gpu(self, id=0):
        return 'cuda:%u' % id

    @property
    def int32(self):
        return torch.int32

    @property
    def float32(self):
        return torch.float32

    def _placeholder(self, dtype=None, shape=None, name=None):
        return TorchPlaceholder(dtype, shape=shape, name=name, device=self.get_current_device())

    def _variable(self, initial_value=None, trainable=True, name=None):
        if isinstance(initial_value, np.ndarray):
            initial_value = torch.from_numpy(initial_value)
        transferred = initial_value.to(self.get_current_device())
        transferred.requires_grad = True
        return transferred

    def _device(self, name):
        return torch.device(name)

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
        return torch.tensor(x, dtype=dtype)

    def cast(self, x, dtype):
        return x.type(dtype)

    def dtype(self, x):
        return x.dtype

    def shape(self, x):
        return x.shape

    def rank(self, x):
        return len(x.shape)

    def abs(self, x):
        return x.abs()

    def set_value(self, x, value):
        raise NotImplementedError

    def floatx(self, as_string=False):
        if as_string:
            return super(PyTorchBackend, self).floatx()
        return getattr(torch, super(PyTorchBackend, self).floatx())

    def zeros(self, shape, dtype=None, name=None):
        dtype = dtype or self.floatx()
        if not isinstance(shape, int):
            shape = tuple(shape)
        return torch.zeros(shape, dtype=dtype)

    def zeros_like(self, x, dtype=None, name=None):
        return torch.zeros_like(x, dtype=dtype)

    def ones(self, shape, dtype=None, name=None):
        dtype = dtype or self.floatx()
        return torch.ones(shape, dtype=dtype)

    def ones_like(self, x, dtype=None, name=None):
        return torch.ones_like(x, dtype=dtype)

    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        dtype = dtype or self.floatx()
        shape = list(shape)
        return mean + stddev * torch.randn(shape, dtype=dtype)

    def random_truncated_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        raise NotImplementedError

    def random_uniform(self, shape, minval=1.0, maxval=None, dtype=None, seed=None):
        dtype = dtype or self.floatx()
        if maxval is None:
            minval, maxval = 0.0, minval
        shape = list(shape)
        samples = torch.rand(shape, dtype=dtype)
        return samples * (maxval - minval) + minval

    def random_binomial(self, shape, p=0.5, dtype=None):
        raise NotImplementedError

    def random_gamma(self, shape, alpha, beta=None):
        raise NotImplementedError

    def tanh(self, x, name=None):
        return torch.tanh(x, name=name)

    def sigmoid(self, x):
        return torch.sigmoid(x)

    def relu(self, x, alpha=0.):
        return torch.relu(x)

    def softmax(self, x, T=1.0):
        return torch.nn.softmax(x)

    def dropout(self, x, p, seed=None):
        raise NotImplementedError

    def conv2d(self, x, kernel, strides=(1, 1), border_mode='same',
               image_shape=None, filter_shape=None):
        x = x.permute(0, 3, 1, 2)
        kernel = kernel.permute(3, 2, 0, 1)

        input_rows = x.size(2)
        input_cols = x.size(3)
        filter_rows = kernel.size(2)
        filter_cols = kernel.size(3)

        out_rows = (input_rows + strides[0] - 1) // strides[0]
        padding_rows = max(0, (out_rows - 1) * strides[0] +
                                (filter_rows - 1) + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        out_cols = (input_cols + strides[0] - 1) // strides[0]
        padding_cols = max(0, (out_cols - 1) * strides[0] +
                                (filter_cols - 1) + 1 - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            x = torch.nn.functional.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        x = torch.nn.functional.conv2d(x, kernel, stride=strides)
        x = x.permute(0, 2, 3, 1)
        return x

    def conv2d_transpose(self, x, kernel, dim_out, strides=(1, 1), border_mode='same'):
        raise NotImplementedError

    def pool2d(self, x, pool_size, strides=(1, 1),
               border_mode='valid', pool_mode='max'):
        '''
        pool_size: tuple of 2 integers.
        strides: tuple of 2 integers.
        border_mode: one of "valid", "same".
        dim_ordering: one of "th", "tf".
        '''
        x = x.permute(0, 3, 1, 2)

        input_rows = x.size(2)
        input_cols = x.size(3)
        filter_rows = pool_size[0]
        filter_cols = pool_size[1]

        out_rows = (input_rows + strides[0] - 1) // strides[0]
        padding_rows = max(0, (out_rows - 1) * strides[0] +
                                (filter_rows - 1) + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        out_cols = (input_cols + strides[0] - 1) // strides[0]
        padding_cols = max(0, (out_cols - 1) * strides[0] +
                                (filter_cols - 1) + 1 - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            x = torch.nn.functional.pad(x, [0, int(cols_odd), 0, int(rows_odd)])

        if pool_mode == 'max':
            x = torch.nn.functional.max_pool2d(x, pool_size, stride=strides)
        elif pool_mode == 'avg':
            x = torch.nn.functional.avg_pool2d(x, pool_size, stride=strides)
        else:
            raise NotImplementedError
        x = x.permute(0, 2, 3, 1)
        return x


    def flatten(self, x, leading=1):
        raise NotImplementedError

    def split(self, x, num_splits, axis=None):
        raise NotImplementedError

    def reshape(self, x, shape):
        if not isinstance(x, int):
            shape = tuple(shape)
        return x.reshape(shape)

    def sum(self, x, axis=None, keepdims=False):
        return inputs.sum(dim=axis, keepdim=keepdims)

    def prod(self, x, axis=None, keepdims=False):
        return inputs.prod(dim=axis, keepdim=keepdims)

    def mean(self, x, axis=None, keepdims=False):
        return inputs.mean(dim=axis, keepdim=keepdims)

    def batch_norm(self, x, beta, gamma):
        raise NotImplementedError

    def log(self, x):
        return torch.log(x)

    def exp(self, x):
        return torch.exp(x)

    def pow(self, x, a):
        return torch.pow(x, a)

    def mul(self, x, y):
        return torch.mul(x, y)

    def sqrt(self, x):
        return torch.sqrt(x)

    def categorical_crossentropy(self, output, target, from_logits=False):
        if from_logits:
            raise NotImplementedError
        output /= output.sum(axis=-1, keepdim=True)
        output = output.clamp(self.epsilon(), 1 - self.epsilon())
        return -(target * torch.log(output)).sum(axis=-1)

    def concatenate(self, tensors, axis=-1):
        return torch.cat(tensors, dim=axis)

    def sort(self, tensor, axis=-1):
        return tensor.sort(dim=axis)

    def argmin(self, tensor, axis=0):
        return tensor.argmin(dim=axis)

    def map(self, function, input):
        return map(function, input)

    def rnn(self, step_function, input, initial_states):
        raise NotImplementedError

    def while_loop(self, condition, body, loop_vars, **kwargs):
        raise NotImplementedError

    def scan(self, fn, elems, initializer=None):
        raise NotImplementedError

    def logdet(self, A, **kwargs):
        A = (A + self.matrix_transpose(A)) / 2.
        term = torch.log(torch.diag(self.cholesky(A, **kwargs)))
        return 2 * term.sum(dim=-1)

    def einsum(self, subscripts, *operands):
        return torch.einsum(subscripts, operands)

    def cholesky(self, A, lower=True, warn=False, correct=True):
        return torch.potrf(A, upper=not lower)

    # Tensorflow interface

    def placeholder(self, dtype, shape=None, name=None):
        return self._placeholder(dtype=dtype, shape=shape, name=name)

    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    def assign(self, a, b):
        raise NotImplementedError

    def to_float(self, x):
        return torch.tensor(x).type(self.floatx())

    def constant(self, value, dtype=None, shape=None):
        return torch.from_numpy(np.array(value))

    def get_shape(self, x):
        return list(x.shape)

    def get_value(self, variable):
        return variable.numpy()

    def concat(self, values, axis=-1):
        values = [self.coerce(v, dtype=self.floatx()) for v in values]
        return torch.cat(values, dim=axis)

    def gather(self, params, indices):
        return torch.gather(params, indices)

    def gather_nd(self, params, indices):
        raise NotImplementedError

    def equal(self, x, y):
        return torch.equal(x, y)

    def logical_and(self, x, y):
        return x and y

    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        if transpose_a:
            a = self.matrix_transpose(a)
        if transpose_b:
            b = self.matrix_transpose(b)
        return torch.matmul(a, b)

    def trace(self, a):
        return torch.trace(a)

    def transpose(self, a, perm=None):
        return a.permute(*perm)

    def matrix_transpose(self, a):
        return torch.transpose(a, -1, -2)

    def matrix_diag(self, a):
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
        return a.inverse()

    def expand_dims(self, x, dim=-1):
        return x.unsqueeze(dim)

    def tile(self, input, multiples):
        return input.repeat(*map(int, multiples.numpy()))

    def gradients(self, loss, variables):
        solo = False
        if not isinstance(variables, list):
            solo = True
        if solo:
            variables = [variables]
        [v.grad.zero_() if v.grad is not None else None for v in variables]
        loss.backward()
        result = [v.grad for v in variables]
        if solo:
            return result[0]
        return result

    def square(self, x):
        return torch.pow(x, 2)

    def clip_by_value(self, x, low, high):
        return x.clamp(low, high)

    def stack(self, values, axis=0, name='stack'):
        return torch.stack(values, dim=axis)

    def unstack(self, values, num=None, axis=0):
        return torch.unbind(values, dim=axis)

    def pack(self, *args, **kwargs):
        return self.stack(*args, **kwargs)

    def unpack(self, *args, **kwargs):
        return self.unstack(*args, **kwargs)

    def reduce_max(self, x, axis=None, keepdims=False):
        return x.max(dim=axis, keepdim=keepdims)

    def reduce_logsumexp(self, x, axis=None, keepdims=False):
        raise NotImplementedError

    def matrix_solve(self, matrix, rhs, adjoint=None):
        import ipdb; ipdb.set_trace()
        return torch.gesv(rhs, matrix)[0]

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
        return x.mm(y)

    def outer(self, x, y):
        if len(self.get_shape(x)) == 0:
            return x * y
        return x[...,:,None] * y[...,None,:]

    def eye(self, d, batch_shape=None):
        out = torch.eye(d)
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
        return self.argmax(x, axis=axis)

    def max(self, x, axis=None, keepdims=False):
        return self.reduce_max(x, axis=axis, keepdims=keepdims)

    def logsumexp(self, x, axis=None, keepdims=False):
        return self.reduce_logsumexp(x, axis=axis, keepdims=keepdims)

    def switch(self, condition, then_expression, else_expression):
        raise NotImplementedError

    def alloc(self, value, shape, unbroadcast=None, dtype=None):
        raise NotImplementedError

    def range(self, start, limit=None, delta=1):
        return torch.arange(start, limit, step=delta)

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
