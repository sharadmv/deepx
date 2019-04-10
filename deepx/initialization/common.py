import math

from deepx.initialization.initializer import Initializer
from deepx.backend import T
from deepx.initialization import util

__all__ = [
    'zeros',
    'ones',
    'normal',
    'truncated_normal',
    'glorot_uniform',
    'glorot_normal',
    'he_uniform',
    'he_normal',
    'lecun_uniform',
    'lecun_normal',
]

class Zeros(Initializer):

    def sample(self):
        return T.zeros(self.shape, dtype=self.dtype)
zeros = Zeros

class Ones(Initializer):

    def sample(self):
        return T.ones(self.shape, dtype=self.dtype)
ones = Ones

class Normal(Initializer):

    def __init__(self, *args, **kwargs):
        self.mean, self.stddev= kwargs.pop('mean', 0), kwargs.pop('stddev', 1)
        super(Normal, self).__init__(*args, **kwargs)

    def sample(self):
        return T.random_normal(self.shape, mean=self.mean, stddev=self.stddev, dtype=self.dtype, seed=self.seed)
normal = Normal

class TruncatedNormal(Normal):

    def sample(self):
        return T.random_truncated_normal(self.shape, mean=self.mean, stddev=self.stddev, dtype=self.dtype, seed=self.seed)
truncated_normal = TruncatedNormal


class VarianceScaling(Initializer):
    """
    Taken from Keras's variance scaling, but uses DeepX backend
    instead
    """

    def __init__(self, *args, **kwargs):
        scale = kwargs.pop('scale', 1.0)
        mode = kwargs.pop('mode', 'fan_in')
        distribution = kwargs.pop('distribution', 'normal')
        super(VarianceScaling, self).__init__(*args, **kwargs)

        if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)
        mode = mode.lower()
        if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
            raise ValueError('Invalid `mode` argument: '
                             'expected on of {"fan_in", "fan_out", "fan_avg"} '
                             'but got', mode)
        distribution = distribution.lower()
        if distribution not in {'normal', 'uniform'}:
            raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)
        self.scale = scale
        self.mode = mode
        self.distribution = distribution

    def sample(self):
        shape = self.shape
        fan_in, fan_out = util.compute_fans(shape)
        scale = self.scale
        if self.mode == 'fan_in':
            scale /= max(1., fan_in)
        elif self.mode == 'fan_out':
            scale /= max(1., fan_out)
        else:
            scale /= max(1., float(fan_in + fan_out) / 2)
        if self.distribution == 'normal':
            stddev = math.sqrt(scale) / .87962566103423978
            return T.random_truncated_normal(shape, 0., stddev,
                                      dtype=self.dtype, seed=self.seed)
        else:
            limit = math.sqrt(3. * scale)
            return T.random_uniform(shape, -limit, limit,
                                    dtype=self.dtype, seed=self.seed)

def lecun_uniform(shape, dtype=None, seed=None):
    """LeCun uniform initializer.

    Taken from Keras
    """
    return VarianceScaling(shape, dtype=dtype,
                           scale=1.,
                           distribution='uniform',
                           seed=seed)


def glorot_normal(shape, dtype=None, seed=None):
    """Glorot normal initializer, also called Xavier normal initializer.

    Taken from Keras
    """
    return VarianceScaling(shape, dtype=dtype,
                           scale=1.,
                           mode='fan_avg',
                           distribution='normal',
                           seed=seed)


def glorot_uniform(shape, dtype=None, seed=None):
    """Glorot uniform initializer, also called Xavier uniform initializer.

    Taken from Keras
    """
    return VarianceScaling(shape, dtype=dtype,
                           scale=1.,
                           mode='fan_avg',
                           distribution='uniform',
                           seed=seed)


def he_normal(shape, dtype=None, seed=None):
    """He normal initializer.

    Taken from Keras
    """
    return VarianceScaling(shape, dtype=dtype,
                           scale=2.,
                           mode='fan_in',
                           distribution='normal',
                           seed=seed)


def lecun_normal(shape, dtype=None, seed=None):
    """LeCun normal initializer.

    Taken from Keras
    """
    return VarianceScaling(shape, dtype=dtype,
                           scale=1.,
                           mode='fan_in',
                           distribution='normal',
                           seed=seed)


def he_uniform(shape, dtype=None, seed=None):
    """He uniform variance scaling initializer.

    Taken from Keras
    """
    return VarianceScaling(shape, dtype=dtype,
                           scale=2.,
                           mode='fan_in',
                           distribution='uniform',
                           seed=seed)
