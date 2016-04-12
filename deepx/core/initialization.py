import numpy as np

def normal(shape, scale=0.01):
    return np.random.normal(loc=0.0, scale=scale, size=shape)

def uniform(shape, a=-0.05, b=0.05):
    return np.random.uniform(a, b, size=shape)

lls = locals()

class Initializer(object):

    def __init__(self):
        self.default = 'uniform'

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
