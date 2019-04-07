import tensorflow_probability as tfp
tfd = tfp.distributions

from deepx.backend import T
from deepx.stats.base import Distribution

__all__ = [
    'Normal',
    'Gaussian',
    'GaussianDiag',
    'Bernoulli',
    'Categorical',
    'Dirichlet',
    'kl_divergence'
]

class TFPWrapper(Distribution):

    def __init__(self, *args, **kwargs):
        self.params = (args, kwargs)
        self.dist = self.driver(*args, **kwargs)

    def sample(self, num_samples=[]):
        return self.dist.sample(num_samples)

    def log_likelihood(self, x):
        return self.dist.log_prob(x)

    def mean(self):
        return self.dist.mean()

    def entropy(self):
        return self.dist.entropy()

class Normal(TFPWrapper):

    def driver(self, mean, std):
        return tfd.Normal(
            loc=mean,
            scale=std,
        )

class Gaussian(TFPWrapper):

    def driver(self, mean, cov):
        return tfd.MultivariateNormalFullCovariance(
            loc=mean,
            covariance_matrix=cov,
        )

class GaussianDiag(TFPWrapper):

    def driver(self, mean, scale_diag):
        return tfd.MultivariateNormalDiag(
            loc=mean,
            scale_diag=scale_diag,
        )

class Dirichlet(TFPWrapper):

    def driver(self, alpha):
        return tfd.Dirichlet(
            concentration=alpha
        )

class Categorical(TFPWrapper):

    def driver(self, probs):
        return tfd.Categorical(
            probs=probs
        )

class Bernoulli(TFPWrapper):

    def driver(self, probs=None, logits=None):
        return tfd.Bernoulli(
            probs=probs,
            logits=logits
        )

def kl_divergence(p, q):
    return tfd.kl_divergence(p.dist, q.dist)
