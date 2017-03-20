from .. import T
import numpy as np

def normal(shape, **kwargs):
    if isinstance(shape, int):
        shape = [shape]
    return T.random_normal(shape, **kwargs)

def uniform(shape, **kwargs):
    if isinstance(shape, int):
        shape = [shape]
    return T.random_uniform(shape, **kwargs)

def xavier(shape, val=0.1, constant=1, **kwargs):
    if isinstance(shape, int):
        shape = [shape]
    if len(shape) == 1:
        return uniform(shape, **kwargs)
    fan_in, fan_out = shape
    low = -constant*np.sqrt(val/(fan_in + fan_out))
    high = constant*np.sqrt(val/(fan_in + fan_out))
    return T.random_uniform((fan_in, fan_out), minval=low, maxval=high)

lls = locals()

class Initializer(object):

    def __init__(self):
        self.default = 'xavier'

    def init(self, shape, init_type, value=None, **kwargs):
        if value is not None:
            return (np.ones(shape) * value).astype(T.floatx())
        if init_type == 'default':
            init_type = self.default
        return lls[init_type](shape, **kwargs)

initializer = Initializer()

def set_default(default):
    initializer.default = default

def initialize_weights(*args, **kwargs):
    return initializer.init(*args, **kwargs)
