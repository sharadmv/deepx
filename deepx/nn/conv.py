import numpy as np
import math

from .. import backend as T

from ..node import Node

class Conv(Node):
    def __init__(self, shape_in, kernel=None, stride=1, pool_factor=2, border_mode="same"):
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
        if self.border_mode == "same":
            h_out = h_in
            w_out = w_in
        elif self.border_mode == "valid":
            h_out = h_in - kernel_height + 1
            w_out = w_in - kernel_width + 1
        else:
            raise Exception("Border mode must be {same, valid}.")

        h_out = int(math.ceil(h_out/float(self.pool_factor)))
        w_out = int(math.ceil(w_out/float(self.pool_factor)))

        return channels_out, h_out, w_out

    def rectify(self, X):
        return T.relu(X)

    def _forward(self, X):
        lin     = T.conv2d(X, self.W, border_mode='same') + T.expand_dims(T.expand_dims(T.expand_dims(self.b, 0), 2), 3)
        act     = self.rectify(lin)
        pooled  = T.pool2d(act, (self.pool_factor, self.pool_factor), strides=(self.pool_factor, self.pool_factor))
        return pooled
