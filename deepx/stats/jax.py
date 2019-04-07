import jax.random as random
import jax.numpy as np
from jax import lax

from deepx.backend import T
from deepx.stats.base import Distribution

__all__ = []

class JaxDistribution(Distribution):

    def __init__(self):
        self.rng = T.rng
