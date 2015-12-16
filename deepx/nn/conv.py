import numpy as np
import math
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

from ..node import Node

class Conv(Node):
    def __init__(self, shape_in, kernel=None, stride=1, pool_factor=2, border_mode="full"):
        super(Conv, self).__init__()

        if kernel is None:
            self.shape_weights = shape_in
        else:
            self.shape_in = shape_in
            self.shape_weights = kernel

        self.border_mode = border_mode
        self.stride = stride
        self.pool_factor = pool_factor


    def initialize(self):
        channels_out, kernel_height, kernel_width = self.shape_weights
        self.W = self.init_parameter('W', (channels_out, self.channels_in, kernel_height, kernel_width))
        self.b = self.init_parameter('b', self.channels_out)

    def _infer(self, shape_in):
        self.channels_in = shape_in[0]
        channels_out, kernel_height, kernel_width = self.shape_weights
        self.channels_out = channels_out
        d_in, h_in, w_in = shape_in
        if self.border_mode == "full":
            h_out = h_in + kernel_height - 1
            w_out = w_in + kernel_width - 1

        elif self.border_mode == "valid":
            h_out = h_in - kernel_height + 1
            w_out = w_in - kernel_width + 1
        else:
            raise Exception("Border mode must be {full, valid}.")

        h_out = int(math.ceil(h_out/2.))
        w_out = int(math.ceil(w_out/2.))


        return channels_out, h_out, w_out

    def rectify(self, X):
        return X * (X > 0)

    def _forward(self, X):
        lin     = conv2d(X, self.W, border_mode='full') + self.b.dimshuffle('x', 0, 'x', 'x')
        act     = self.rectify(lin)
        pooled  = max_pool_2d(act, (self.pool_factor, self.pool_factor), ignore_border=False)
        return pooled

class Reshape(Node):

    def __init__(self, shape_in, shape_out=None):
        super(Reshape, self).__init__()
        if shape_out is None:
            self.shape_in = None
            self.shape_out = shape_in
        else:
            self.shape_in = shape_in
            self.shape_out = shape_out

    def _infer(self, shape_in):
        return self.shape_out

    def _forward(self, X):
        N = X.shape[0]
        shape_out = self.shape_out
        if not isinstance(shape_out, tuple):
            shape_out = (self.shape_out,)
        return X.reshape((N,) + shape_out)

class Flatten(Node):

    def _infer(self, shape_in):
        return np.product(shape_in)

    def _forward(self, X):
        N = X.shape[0]
        return X.reshape((N, -1))

    def to_str(self):
        return "Flatten(%s, %s)" % (self.shape_in, self.shape_out)
