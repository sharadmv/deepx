from .. import T

from .common import ExponentialFamily

__all__ = ["InverseWishart", "IW"]

class InverseWishart(ExponentialFamily):

    def get_param_dim(self):
        return [2, 0]

    def sample(self, num_samples=1):
        S, nu = self.get_parameters('regular')
        leading_dim = T.shape(S)[:-2]
        dim = T.shape(S)[-1]
        nu_diag = T.to_float(T.range(dim))
        leading_shape = T.concatenate([[num_samples], leading_dim])
        chol = T.tile(T.cholesky(T.matrix_inverse(S))[None], T.concatenate([[num_samples], T.cast(T.ones(T.rank(S)), T.int32)]))
        random_variates = T.random_normal(T.concatenate([leading_shape, [dim, dim]]))
        gamma_param = nu - nu_diag
        diagonal = T.sqrt(T.random_gamma(leading_shape, gamma_param / 2., 0.5))
        random_variates = T.band_part(T.set_diag(random_variates, diagonal), -1, 0)
        random_variates = T.matrix_inverse(chol@random_variates@T.matrix_transpose(random_variates)@T.matrix_transpose(chol))
        return random_variates

    def log_likelihood(self, x):
        raise NotImplementedError

    def expected_value(self):
        S, nu = self.get_parameters('regular')
        D = T.to_float(T.shape(S)[-1])
        return S / (nu[..., None, None] - D - 1)

    def sufficient_statistics(self, sigma):
        sigma_inv = T.matrix_inverse(sigma)
        return [
            -1 / 2. * sigma_inv,                 # -\frac{1}{2} \Sigma^{-1}
            -1 / 2. * T.logdet(sigma)              # -\frac{1}{2} \log |\Sigma|
        ]

    def expected_sufficient_statistics(self, fudge=1e-8):
        return T.grad(self.log_z(), self.get_parameters('natural'))
        S, nu = self.get_parameters('regular')
        d = T.shape(S)[-1]
        S_inv = T.matrix_inverse(S)
        g1 = -0.5 * nu[..., None, None] * S_inv
        g2 = 0.5 * (T.cast(d, T.dtype(S)) * T.cast(T.log(2.), T.dtype(S))
                     - T.logdet(S)
                     + T.sum(T.digamma((nu[...,None] - T.cast(T.range(d), T.dtype(S)))/2.), -1))
        return [g1, g2]

    def log_h(self, x):
        return 1

    def log_z(self):
        S, nu = self.natural_to_regular(self.get_parameters('natural'))
        d = T.to_float(T.shape(S)[-1])
        return (nu / 2. * (d * T.log(2.) - T.logdet(S))
                + T.multigammaln(nu / 2., d))

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        S, nu = regular_parameters
        d = T.cast(T.shape(S)[-1], T.dtype(S))
        return [S, nu + d + 1]

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        A, b = natural_parameters
        d = T.cast(T.shape(A)[-1], T.dtype(A))
        return [A, b - d - 1]

IW = InverseWishart
