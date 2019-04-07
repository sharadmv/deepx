import torch

from deepx.backend import T
from deepx.stats.base import Distribution

__all__ = [
    'Gaussian',
    'GaussianDiag',
    'Normal',
    'Bernoulli',
    'Categorical',
    'Dirichlet',
    'kl_divergence'
]

class PyTorchWrapper(Distribution):

    def __init__(self, *args, **kwargs):
        self.params = (args, kwargs)
        self.dist = self.driver(*args, **kwargs)

    def sample(self, num_samples=torch.Size([])):
        if isinstance(num_samples, int):
            num_samples = [num_samples]
        if self.dist.has_rsample:
            return self.dist.rsample(num_samples)
        return self.dist.sample(num_samples)

    def log_likelihood(self, x):
        return self.dist.log_prob(x)

    def mean(self):
        return self.dist.mean()

    def entropy(self):
        return self.dist.entropy()

class Normal(PyTorchWrapper):

    def driver(self, mean, std):
        return torch.distributions.Normal(
            loc=mean,
            scale=std
        )

class Gaussian(PyTorchWrapper):

    def driver(self, mean, cov):
        return torch.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=cov,
        )

class GaussianDiag(PyTorchWrapper):

    def driver(self, mean, cov_diag):
        return torch.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=T.matrix_diag(cov_diag),
        )


class Dirichlet(PyTorchWrapper):

    def driver(self, alpha):
        return torch.distributions.Dirichlet(
            concentration=alpha
        )

class Categorical(PyTorchWrapper):

    def driver(self, probs):
        return torch.distributions.Categorical(
            probs=probs
        )

class Bernoulli(PyTorchWrapper):

    def driver(self, probs=None, logits=None):
        return torch.distributions.Bernoulli(
            probs=probs,
            logits=logits
        )

def kl_divergence(p, q):
    return torch.distributions.kl_divergence(p.dist, q.dist)
