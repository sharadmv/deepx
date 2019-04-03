import math
import numpy as np

from deepx.backend import T
from deepx.core import Layer
from deepx.nn.activations import Relu

class Convolution(Layer):

    def __init__(self, kernel=(1, 2, 2), border_mode='same', strides=(1, 1)):
        super(Convolution, self).__init__()
        self.kernel = kernel
        self.border_mode = border_mode
        self.strides = strides
        self.in_channels = None
        self.dim_in = self.dim_out = None
        self.initialized = False

    def initialize(self):
        kernel_height, kernel_width, out_channels = self.kernel
        self.create_parameter('W', [kernel_height, kernel_width, self.in_channels, out_channels])
        self.create_parameter('b', [out_channels])

    def is_initialized(self):
        return self.initialized

    def shape_inference(self):
        shape_in = self.get_shape_in()
        if shape_in is None: return
        assert len(shape_in) == 1
        shape_in = shape_in[0]
        self.dim_in = shape_in[-3:]
        self.in_channels = self.dim_in[-1]
        kernel_height, kernel_width, out_channels = self.kernel
        in_height, in_width = self.dim_in[:2]
        if self.border_mode == 'same':
            out_height = int(math.ceil(float(in_height) / float(self.strides[0])))
            out_width = int(math.ceil(float(in_width) / float(self.strides[1])))
        elif self.border_mode == 'valid':
            out_height = int(math.ceil(float(in_height - kernel_height + 1) / float(self.strides[0])))
            out_width  = int(math.ceil(float(in_width - kernel_width + 1) / float(self.strides[1])))
        else:
            raise Exception("Border mode must be {same, valid}.")
        self.dim_out = [out_height, out_width, out_channels]
        self.set_shape_out([shape_in[:-3] + self.dim_out])
        if not self.is_initialized():
            self.initialize()
            self.initialized = True

    def _forward(self, X):
        W, b = self.get_parameter("W"), self.get_parameter("b")
        return (T.conv2d(X, W, border_mode=self.border_mode, strides=self.strides)
                + b[None, None, None])

    def __repr__(self):
        return "Convolution(%s, %s)" % (self.dim_in, self.dim_out)

class Pool(Layer):

    def __init__(self, kernel=(2, 2), strides=(2, 2), pool_type='max', border_mode='same'):
        super(Pool, self).__init__()
        self.kernel = kernel
        self.strides = strides
        self.pool_type = pool_type
        self.border_mode = border_mode
        self.dim_in = self.dim_out = None
        self.initialized = False

    def initialize(self):
        pass

    def is_initialized(self):
        return True

    def shape_inference(self):
        shape_in = self.get_shape_in()
        if shape_in is None: return
        assert len(shape_in) == 1
        shape_in = shape_in[0]
        self.dim_in = shape_in[-3:]
        kernel_height, kernel_width = self.kernel
        in_height, in_width = self.dim_in[:2]
        if self.border_mode == 'same':
            out_height = int(math.ceil(float(in_height) / float(self.strides[0])))
            out_width = int(math.ceil(float(in_width) / float(self.strides[1])))
        elif self.border_mode == 'valid':
            out_height = int(math.ceil(float(in_height - kernel_height + 1) / float(self.strides[0])))
            out_width  = int(math.ceil(float(in_width - kernel_width + 1) / float(self.strides[1])))
        else:
            raise Exception("Border mode must be {same, valid}.")
        self.dim_out = [out_height, out_width, self.dim_in[-1]]
        self.set_shape_out([shape_in[:-3] + self.dim_out])
        if not self.is_initialized():
            self.initialize()
            self.initialized = True

    def _forward(self, X, **kwargs):
        return T.pool2d(X, self.kernel, strides=self.strides, border_mode=self.border_mode)

    def __repr__(self):
        return "Pool(%s, %s)" % (self.dim_in, self.dim_out)

class Deconvolution(Layer):

    def __init__(self, kernel, border_mode='same', strides=(2, 2)):
        super(Deconvolution, self).__init__()
        self.kernel = kernel
        self.border_mode = border_mode
        self.strides = strides
        self.in_channels = None
        self.dim_in = self.dim_out = None
        self.initialized = False

    def initialize(self):
        kernel_height, kernel_width, out_channels = self.kernel
        self.create_parameter('W', [kernel_height, kernel_width, out_channels, self.in_channels])
        self.create_parameter('b', [out_channels])

    def is_initialized(self):
        return not (self.dim_in is None or self.dim_out is None)

    def shape_inference(self):
        shape_in = self.get_shape_in()
        if shape_in is None: return
        assert len(shape_in) == 1
        shape_in = shape_in[0]
        self.dim_in = shape_in[-3:]
        self.in_channels = self.dim_in[-1]
        kernel_height, kernel_width, out_channels = self.kernel
        in_height, in_width = self.dim_in[:2]
        if self.border_mode == 'same':
            out_height = int(in_height * self.strides[0])
            out_width = int(in_width * self.strides[1])
        elif self.border_mode == 'valid':
            raise NotImplementedError
        else:
            raise Exception("Border mode must be {same, valid}.")
        self.dim_out = [out_height, out_width, out_channels]
        self.set_shape_out([shape_in[:-3] + self.dim_out])
        if not self.is_initialized():
            self.initialize()
            self.initialized = True

    def _forward(self, X):
        W, b = self.get_parameter("W"), self.get_parameter("b")
        result = (T.conv2d_transpose(X, W, self.dim_out, border_mode=self.border_mode, strides=self.strides)
                + b[None, None, None])
        return result

    def __repr__(self):
        return "Deconvolution(%s, %s)" % (self.dim_in, self.dim_out)

class SpatialSoftmax(Layer):

    def __init__(self):
        super(SpatialSoftmax, self).__init__()
        self.x_map = self.y_map = None
        self.dim_in = self.dim_out = None
        self.initialized = False

    def initialize(self):
        h, w = self.in_height, self.in_width
        x_map = np.empty((h, w))
        y_map = np.empty((h, w))

        for i in range(self.in_height):
            for j in range(self.in_width):
                x_map[i, j], y_map[i, j] = (i - h / 2.0) / h, (j - w / 2.0) / w

        self.x_map = T.reshape(T.constant(x_map, dtype=T.floatx()), [h * w])
        self.y_map = T.reshape(T.constant(y_map, dtype=T.floatx()), [h * w])

    def is_initialized(self):
        return not (self.dim_in is None or self.dim_out is None)

    def __getstate__(self):
        state = super(SpatialSoftmax, self).__getstate__()
        state.pop('x_map')
        state.pop('y_map')
        return state

    def shape_inference(self):
        shape_in = self.get_shape_in()
        if shape_in is None: return
        assert len(shape_in) == 1
        shape_in = shape_in[0]
        self.dim_in = self.in_height, self.in_width, self.in_channels = shape_in[-3:]
        self.dim_out = [self.in_channels * 2]
        self.set_shape_out([shape_in[:-3] + self.dim_out])
        if not self.is_initialized():
            self.initialize()
            self.initialized = True

    def _forward(self, X):
        h, w, c = self.in_height, self.in_width, self.in_channels
        features = T.reshape(T.transpose(X, perm=[0, 3, 1, 2]), [-1, h * w])
        softmax = T.softmax(features)
        fp_x = T.sum(T.mul(self.x_map, softmax), axis=[1], keepdims=True)
        fp_y = T.sum(T.mul(self.y_map, softmax), axis=[1], keepdims=True)
        return T.reshape(T.concat([fp_x, fp_y], axis=1), [-1, c * 2])

    def __repr__(self):
        return "Spatial(%s, %s)" % (self.dim_in, self.dim_out)

def Conv(conv_kernel, pool_kernel=(2, 2), pool_strides=(2, 2), border_mode='same', pool_type='max', activation=Relu):
    return Convolution(conv_kernel, border_mode=border_mode) >> activation() >> Pool(kernel=pool_kernel, strides=pool_strides)

def Deconv(deconv_kernel, border_mode='same', strides=(2, 2), activation=Relu):
    return Deconvolution(deconv_kernel, border_mode=border_mode, strides=strides) >> activation()

def SpatialConv(conv_kernel, pool_kernel=(2, 2), pool_strides=(2, 2), border_mode='same', pool_type='max', activation=Relu):
    return Convolution(conv_kernel, border_mode=border_mode) >> activation() >> SpatialSoftmax()
