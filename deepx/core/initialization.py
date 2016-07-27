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

lls = locals()

class Initializer(object):

    def __init__(self):
        self.default = 'normal'

    def init(self, shape, init_type, value=None, **kwargs):
        if value is not None:
            return np.ones(shape) * value
        if init_type == 'default':
            init_type = self.default
        return lls[init_type](shape, **kwargs)

initializer = Initializer()

def set_default(default):
    initializer.default = default

def initialize_weights(*args, **kwargs):
    return initializer.init(*args, **kwargs)
