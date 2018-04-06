import math

from .. import T
from ..layer import Layer
from .full import Relu

class Convolution(Layer):

    def __init__(self, kernel=(1, 2, 2), border_mode='same', strides=(1, 1)):
        super(Convolution, self).__init__()
        self.kernel = kernel
        self.border_mode = border_mode
        self.strides = strides
        self.in_channels = None
        self.dim_in = self.dim_out = None

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def initialize(self):
        kernel_height, kernel_width, out_channels = self.kernel
        self.create_parameter('W', [kernel_height, kernel_width, self.in_channels, out_channels])
        self.create_parameter('b', [out_channels])

    def is_initialized(self):
        return not (self.dim_in is None or self.dim_out is None)

    def infer_shape(self, shape):
        if shape is None: return
        self.dim_in = shape[-3:]
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

    def forward(self, X, **kwargs):
        W, b = self.get_parameter_list('W', 'b')
        return (T.conv2d(X, W, border_mode=self.border_mode, strides=self.strides)
                + b[None, None, None])

    def __str__(self):
        return "Convolution(%s, %s)" % (self.dim_in, self.dim_out)

class Pool(Layer):

    def __init__(self, kernel=(2, 2), strides=(2, 2), pool_type='max', border_mode='same'):
        super(Pool, self).__init__()
        self.kernel = kernel
        self.strides = strides
        self.pool_type = pool_type
        self.border_mode = border_mode
        self.dim_in = self.dim_out = None

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def initialize(self):
        pass

    def is_initialized(self):
        return True

    def infer_shape(self, shape):
        if shape is None: return
        self.dim_in = shape[-3:]
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

    def forward(self, X, **kwargs):
        return T.pool2d(X, self.kernel, strides=self.strides, border_mode=self.border_mode)

    def __str__(self):
        return "Pool(%s, %s)" % (self.dim_in, self.dim_out)

class Deconvolution(Layer):

    def __init__(self, kernel, border_mode='same', strides=(2, 2)):
        super(Deconvolution, self).__init__()
        self.kernel = kernel
        self.border_mode = border_mode
        self.strides = strides
        self.in_channels = None
        self.dim_in = self.dim_out = None

    def get_dim_in(self):
        return self.dim_in

    def get_dim_out(self):
        return self.dim_out

    def initialize(self):
        kernel_height, kernel_width, out_channels = self.kernel
        self.create_parameter('W', [kernel_height, kernel_width, out_channels, self.in_channels])
        self.create_parameter('b', [out_channels])

    def is_initialized(self):
        return not (self.dim_in is None or self.dim_out is None)

    def infer_shape(self, shape):
        if shape is None: return
        self.dim_in = shape[-3:]
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

    def forward(self, X, **kwargs):
        W, b = self.get_parameter_list('W', 'b')
        result = (T.conv2d_transpose(X, W, self.dim_out, border_mode=self.border_mode, strides=self.strides)
                + b[None, None, None])
        return result

    def __str__(self):
        return "Deconvolution(%s, %s)" % (self.dim_in, self.dim_out)

def Conv(conv_kernel, pool_kernel=(2, 2), pool_strides=(2, 2), border_mode='same', pool_type='max', activation=Relu):
    return Convolution(conv_kernel, border_mode=border_mode) >> activation() >> Pool(kernel=pool_kernel, strides=pool_strides)

def Deconv(deconv_kernel, border_mode='same', strides=(2, 2), activation=Relu):
    return Deconvolution(deconv_kernel, border_mode=border_mode, strides=strides) >> activation()
