import logging
import six
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

def uses_device(method):
    method.uses_device = True
    return method

@six.add_metaclass(ABCMeta)
class FunctionBase(object):

    def __init__(self, session, inputs, outputs, updates, lazy=True):
        self.session = session
        self.inputs = inputs
        self.outputs = outputs
        self.updates = updates
        self.lazy = lazy

    def feed_dict(self, *inputs):
        return {i: input for i, input in zip(self.inputs, inputs)}

    @abstractmethod
    def __call__(self, *inputs):
        pass


@six.add_metaclass(ABCMeta)
class BackendBase(object):
    """
    Base class for DeepX backends.
    Outlines the methods and signatures for a general set of methods
    used in building and utilizing symbolic graphs.
    """

    def __init__(self, use_cudnn=True):
        self._FLOATX = 'float32'
        self._EPSILON = 10e-8
        self._DEFAULT_DEVICE = self.cpu(0)
        self._DEFAULT_INITIALIZATION = 'glorot_uniform'
        self._device_stack = []
        self._initialization_stack = []
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

    def get_default_initialization(self):
        return self._DEFAULT_INITIALIZATION

    def set_default_initialization(self, default_initialization):
        self._DEFAULT_INITIALIZATION = default_initialization

    # General methods

    @classmethod
    @abstractmethod
    def use_device(cls, method):
        """
        use_device(method)
        A decorator that will force the output of a function to
        belong to a certain device context.
        """
        pass


    @abstractmethod
    def cpu(self, id=0):
        pass

    @abstractmethod
    def gpu(self, id=0):
        pass

    @contextmanager
    def device(self, name):
        """
        device(name)
        Assigns a device context to `deepx` functions.

        Args:
            name (str): The name of the device

        Examples:
            Using DeepX backend directly:
                >>> from deepx import T
                >>> with T.device('/gpu:0'):
                ...     X = T.matrix()

            Using DeepX to define a network:
                >>> from deepx import T
                >>> with T.device('/gpu:0'):
                ...     network = Vector(784) >> Tanh(200) >> Softmax(10)
        """
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
        """
        Returns the current device context.

        Returns:
            str: Name of current device
        """
        if len(self._device_stack) == 0:
            return self.get_default_device()
        return self._device_stack[-1]

    @contextmanager
    def initialization(self, initialization):
        """
        initialization(initialization)
        Assigns an initialization context to `deepx` networks.

        Args:
            initialization (str): The name of the device

        """
        self.push_initialization(initialization)
        yield
        self.pop_initialization()

    def push_initialization(self, initialization):
        self._initialization_stack.append(initialization)
        logging.debug("Pushed initialization onto stack: {initialization}".format(initialization=initialization))

    def pop_initialization(self):
        initialization = self._initialization_stack.pop()
        logging.debug("Popped initialization from stack: {initialization}".format(initialization=initialization))

    def get_current_initialization(self):
        """
        Returns the current initialization context.

        Returns:
            str: Name of current initialization
        """
        if len(self._initialization_stack) == 0:
            return self.get_default_initialization()
        return self._initialization_stack[-1]

    @abstractmethod
    @contextmanager
    def session(self, allow_soft_placement=False, log_device_placement=False):
        """
        session()
        Assigns a session context to `deepx` functions.
        """
        pass

    @abstractmethod
    def interactive_session(self, **kwargs):
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
    def cast(self, x, dtype):
        pass

    @abstractmethod
    def dtype(self, x):
        pass

    @uses_device
    @abstractmethod
    def shape(self, x):
        pass

    @uses_device
    @abstractmethod
    def rank(self, x):
        pass

    @uses_device
    @abstractmethod
    def abs(self, x):
        pass

    @uses_device
    @abstractmethod
    def zeros(self, shape, dtype=None, name=None):
        """
        Creates a tensor filled with zeros.

        Args:
            shape (:obj:`list` of :obj:`int`): The shape of the output tensor.
            dtype (:obj:`str`, optional): The `dtype` of the output tensor.
            name (:obj:`str`, optional): The name of the tensor. Defaults to `None`.

        Returns:
            A tensor filled with zeros with shape `shape`.
        """
        pass

    @uses_device
    @abstractmethod
    def zeros_like(self, x, dtype=None, name=None):
        """
        Creates a tensor filled with zeros.

        Args:
            x: An input tensor.
            dtype (:obj:`str`, optional): The `dtype` of the returned tensor.
            name (:obj:`str`, optional): The name of the returned tensor. Defaults to `None`.

        Returns:
            A tensor filled with zeros that has the same shape as `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def ones(self, shape, dtype=None, name=None):
        """
        Creates a tensor filled with ones.

        Args:
            shape (:obj:`list` of :obj:`int`): The shape of the output tensor.
            dtype (:obj:`str`, optional): The `dtype` of the output tensor.
            name (:obj:`str`, optional): The name of the tensor. Defaults to `None`.

        Returns:
            A tensor filled with ones with shape `shape`.
        """
        pass

    @uses_device
    @abstractmethod
    def ones_like(self, shape, dtype=None, name=None):
        """
        Creates a tensor filled with ones.

        Args:
            x: An input tensor.
            dtype (:obj:`str`, optional): The `dtype` of the returned tensor.
            name (:obj:`str`, optional): The name of the returned tensor. Defaults to `None`.

        Returns:
            A tensor filled with ones that has the same shape as `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def random_normal(self, shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        """
        Returns a random normal tensor initialization.

        Args:
            shape (:obj:`list` of :obj:`int`): The shape of the output tensor initialization.
            mean (:obj:`float`, optional): The mean of the normal distribution.
            stddev (:obj:`float`, optional): The standard deviation of the normal distribution.
            dtype (:obj:`str`, optional): The `dtype` of the returned tensor.
            seed (:obj:`int`, optional): A random seed.

        Returns:
            A tensor initialization with shape `shape` sampled from a normal distribution
            with mean `mean` and standard deviation `stddev`.
        """
        pass

    @uses_device
    @abstractmethod
    def random_uniform(self, shape, minval=0, maxval=None, dtype=None, seed=None):
        """
        Returns a random uniform tensor initialization.

        Args:
            shape (:obj:`list` of :obj:`int`): The shape of the output tensor initialization.
            minval (:obj:`float`, optional): The minimum value of the uniform distribution.
            maxval (:obj:`float`, optional): The maximum value of the uniform distribution.
            dtype (:obj:`str`, optional): The `dtype` of the returned tensor.
            seed (:obj:`int`, optional): A random seed.

        Returns:
            A tensor initialization with shape `shape` sampled from a uniform distribution
            with minimum value `minval` and maximum value `maxval`.
        """
        pass

    @uses_device
    @abstractmethod
    def random_truncated_normal(self, shape, minval=0, maxval=None, dtype=None, seed=None):
        """
        Returns a random truncated_normal tensor initialization.

        Args:
            shape (:obj:`list` of :obj:`int`): The shape of the output tensor initialization.
            minval (:obj:`float`, optional): The minimum value of the uniform distribution.
            maxval (:obj:`float`, optional): The maximum value of the uniform distribution.
            dtype (:obj:`str`, optional): The `dtype` of the returned tensor.
            seed (:obj:`int`, optional): A random seed.

        Returns:
            A tensor initialization with shape `shape` sampled from a uniform distribution
            with minimum value `minval` and maximum value `maxval`.
        """
        pass

    @uses_device
    @abstractmethod
    def random_binomial(self, shape, p=0.5, dtype=None):
        pass

    @uses_device
    @abstractmethod
    def random_gamma(self, shape, alpha, beta=None):
        pass

    @uses_device
    @abstractmethod
    def tanh(self, x, name=None):
        """
        Returns the hyperbolic tangent of a tensor.

        Args:
            x: An input tensor.
            name (:obj:`str`, optional): The name of the returned tensor. Defaults to `None`.

        Returns:
            The tanh applied elementwise to tensor `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def sigmoid(self, x, name=None):
        """
        Returns the sigmoid of a tensor.

        Args:
            x: An input tensor.
            name (:obj:`str`, optional): The name of the returned tensor. Defaults to `None`.

        Returns:
            The sigmoid applied elementwise to tensor `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def relu(self, x, name=None):
        """
        Returns the ReLU of a tensor.

        Args:
            x: An input tensor.
            name (:obj:`str`, optional): The name of the returned tensor. Defaults to `None`.

        Returns:
            The rectified-linear unit applied elementwise to tensor `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def softmax(self, x, T=1.0):
        """
        Returns the softmax of a tensor.

        Args:
            x: An input tensor.
            name (:obj:`str`, optional): The name of the returned tensor. Defaults to `None`.

        Returns:
            The softmax applied elementwise to tensor `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def dropout(self, x, p, seed=None):
        """
        Applies dropout to a tensor.

        Args:
            x: An input tensor.
            p: The dropout probability.
            seed (:obj:`int`, optional): A random seed.

        Returns:
            A tensor with elements set to 0 with probability `p`.
        """
        pass

    @uses_device
    @abstractmethod
    def conv2d(self, x, kernel, strides=(1, 1), border_mode='same',
               image_shape=None, filter_shape=None):
        """
        Applies a 2D convolution to an input tensor.

        Args:
            x: An input tensor with shape `[batch_size, channels, height, width]`.
            kernel: An input tensor with shape `[channels_out, channels_in, height, width]`.
            strides: The stride lengths for the dimensions of the input.
            border_mode: (:obj:`str`, optional): a string from `"same"` or `"valid"`.

        Returns:
            The tensor resulting from convolving
            the kernel over the input.
        """
        pass

    @uses_device
    @abstractmethod
    def pool2d(self, x, pool_size, strides=(1, 1),
               border_mode='valid', pool_mode='max'):
        """
        Applies a 2D pooling operation to an input tensor.

        Args:
            x: An input tensor with shape `[batch_size, channels, height, width]`.
            pool_size: The size of pooling window for each of the two dimensions.
            strides: The stride lengths for the dimensions of the input.
            border_mode: (:obj:`str`, optional): a string from `"same"` or `"valid"`. Defaults
                        to `"valid"`.
            pool_mode: (:obj:`str`, optional): a string from `"max"` or `"avg"`. Defaults
                        to `"max"`.

        Returns:
            The tensor resulting from pooling
            the input.
        """
        pass

    @uses_device
    @abstractmethod
    def flatten(self, x):
        """
        Flatten the inputs along all axes but the first.

        Args:
            x: An input tensor.

        Returns:
            The flattened tensor.
        """
        pass

    @uses_device
    @abstractmethod
    def split(self, axis, num_splits, x):
        pass

    @uses_device
    @abstractmethod
    def reshape(self, x, shape):
        pass

    @uses_device
    @abstractmethod
    def sum(self, x, axis=None, keepdims=False):
        pass

    @abstractmethod
    def prod(self, x, axis=None, keepdims=False):
        pass

    @uses_device
    @abstractmethod
    def mean(self, x, axis=None, keepdims=False):
        """
        Takes the mean of a tensor along a particular axis (or none).

        Args:
            x: An input tensor.
            axis (:obj:`int`, optional): The axis which is reduced. Defaults to all axes.
            keepdims (:obj:`bool`, optional): Keeps the reduced dimensions if True.

        Returns:
            The mean of the tensor along the specified axis.
        """
        pass

    @uses_device
    @abstractmethod
    def log(self, x):
        """
        Takes the elementwise natural logarithm of a tensor.

        Args:
            x: An input tensor.

        Returns:
            The elementwise log of `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def exp(self, x):
        """
        Takes the elementwise exponent of a tensor.

        Args:
            x: An input tensor.

        Returns:
            The elementwise exponent of `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def pow(self, x, a):
        """
        Raises a tensor to a power elementwise.

        Args:
            x: An input tensor.
            a: A number.

        Returns:
            The elementwise power of `x` to `a`.
        """
        pass

    @uses_device
    @abstractmethod
    def sqrt(self, x):
        """
        Takes the elementwise square root of a tensor.

        Args:
            x: An input tensor.

        Returns:
            the elementwise square root of `x`.
        """
        pass

    @uses_device
    @abstractmethod
    def sort(self, tensor):
        pass

    @uses_device
    @abstractmethod
    def argmin(self, tensor, axis=0):
        pass

    @uses_device
    @abstractmethod
    def categorical_crossentropy(self, x, target, from_logits=False):
        """
        Calculates the cross entropy of a distribution
        with relation to a target distribution.

        Args:
            x: An input 2D-tensor of shape `[batch_size, N]`.
            target: An input 2D-tensor of shape `[batch_size, N]`.
            from_logits (:obj:`bool`): If true `x` contains unscaled logits
                                        and if false `x` is softmax probabilities.
        Returns:
            A tensor of shape `[batch_size]` with cross entropy between the distributions.
        """
        pass

    @uses_device
    @abstractmethod
    def concatenate(self, tensors, axis=-1):
        pass

    @uses_device
    @abstractmethod
    def rnn(self, step_function, input, initial_states):
        pass

    @uses_device
    @abstractmethod
    def scan(self, fn, elems, initializer=None):
        pass

    @uses_device
    @abstractmethod
    def while_loop(self, condition, body, loop_vars, **kwargs):
        pass

    @uses_device
    @abstractmethod
    def logdet(self, A):
        pass

    @uses_device
    @abstractmethod
    def einsum(self, subscripts, *operands):
        pass

    @uses_device
    @abstractmethod
    def cholesky(self, A, lower=True):
        pass

    # Tensorflow interface

    @abstractmethod
    def placeholder(self, dtype, shape=None, name=None):
        pass

    @abstractmethod
    def variable(self, initial_value=None, trainable=True, name=None):
        return self._variable(initial_value=initial_value, trainable=trainable, name=name)

    @abstractmethod
    def assign(self, a, b):
        pass

    @uses_device
    @abstractmethod
    def to_float(self, x):
        pass

    @uses_device
    @abstractmethod
    def constant(self, value, dtype=None, shape=None):
        pass

    @abstractmethod
    def get_shape(self, x):
        pass

    @abstractmethod
    def get_value(self, variable):
        pass

    @uses_device
    @abstractmethod
    def concat(self, values, axis=-1):
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
    def equal(self, x, y):
        pass

    @uses_device
    @abstractmethod
    def logical_and(self, x, y):
        pass

    @uses_device
    @abstractmethod
    def matmul(self, a, b, transpose_a=False, transpose_b=False, a_is_sparse=False, b_is_sparse=False, name=None):
        pass

    @uses_device
    @abstractmethod
    def trace(self, a):
        pass

    @uses_device
    @abstractmethod
    def transpose(self, a, perm=None):
        pass

    @uses_device
    @abstractmethod
    def matrix_transpose(self, a):
        pass

    @uses_device
    @abstractmethod
    def matrix_diag(self, a):
        pass

    @uses_device
    @abstractmethod
    def lower_triangular(self, a):
        pass

    @uses_device
    @abstractmethod
    def kronecker(self, a, b):
        pass

    @uses_device
    @abstractmethod
    def matrix_inverse(self, a):
        pass

    @uses_device
    @abstractmethod
    def expand_dims(self, x, dim=-1):
        pass

    @uses_device
    @abstractmethod
    def tile(self, input, multiples):
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

    @uses_device
    @abstractmethod
    def reduce_logsumexp(self, x, axis=None, keepdims=False):
        pass

    @uses_device
    @abstractmethod
    def matrix_solve(self, matrix, rhs, adjoint=None):
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
    def arange(self, start, stop=None, step=None):
        pass

    @uses_device
    @abstractmethod
    def dot(self, x, y):
        pass

    @uses_device
    @abstractmethod
    def outer(self, x, y):
        pass

    @uses_device
    @abstractmethod
    def eye(self, d):
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
    def argmax(self, x, axis=None, keepdims=False):
        pass

    @uses_device
    @abstractmethod
    def max(self, x, axis=None, keepdims=False):
        pass

    @uses_device
    @abstractmethod
    def logsumexp(self, x, axis=None, keepdims=False):
        pass

    @uses_device
    @abstractmethod
    def alloc(self, value, shape, unbroadcast=None):
        pass

    @uses_device
    @abstractmethod
    def range(self, limit, delta=1):
        pass

    @uses_device
    @abstractmethod
    def solve(self, a, b):
        pass

    @uses_device
    @abstractmethod
    def one_hot(self, indices, depth):
        pass

    # Science methods

    @uses_device
    @abstractmethod
    def gammaln(self, x):
        pass

    @uses_device
    @abstractmethod
    def multigammaln(self, a, p):
        pass

    @uses_device
    @abstractmethod
    def digamma(self, a):
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
