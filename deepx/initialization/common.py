import numpy as np
from deepx import T

def initialize_weights(name, shape, value=None, **kwargs):
    return globals()[name](shape, **kwargs)

def get_fans(shape, dim_ordering='th'):
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4 or len(shape) == 5:
        # Assuming convolution kernels (2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if dim_ordering == 'th':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif dim_ordering == 'tf':
            receptive_field_size = np.prod(shape[:2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid dim_ordering: ' + dim_ordering)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out

def uniform(shape, scale=0.05):
    return T.random_uniform(shape, minval=-scale, maxval=scale)

def normal(shape, scale=0.05):
    return T.random_normal(shape, mean=0.0, stddev=scale)

def truncated_normal(shape, scale=0.05):
    return T.random_truncated_normal(shape, mean=0.0, stddev=scale)

def lecun_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    scale = np.sqrt(3. / fan_in)
    return uniform(shape, scale)

def xavier(shape, constant=1):
    """ Xavier initialization of network weights"""
    fan_in, fan_out = get_fans(shape)
    low = -constant*np.sqrt(6.0/(fan_in + fan_out))
    high = constant*np.sqrt(6.0/(fan_in + fan_out))
    return T.random_uniform(shape,
                             minval=low, maxval=high,
                             dtype=T.floatx())

def glorot_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(0.1 / (fan_in + fan_out))
    return T.random_normal(shape, s)


def glorot_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / (fan_in + fan_out))
    return uniform(shape, s)


def he_normal(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(2. / fan_in)
    return normal(shape, s)


def he_uniform(shape):
    fan_in, fan_out = get_fans(shape)
    s = np.sqrt(6. / fan_in)
    return uniform(shape, s)


def orthogonal(shape, scale=1.1):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return T.variable(scale * q[:shape[0], :shape[1]])


def identity(shape, scale=1):
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError('Identity matrix initialization can only be used '
                         'for 2D square matrices.')
    else:
        return T.variable(scale * np.identity(shape[0]))


def zero(shape):
    return T.zeros(shape)


def one(shape):
    return T.ones(shape)
