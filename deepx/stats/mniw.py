from .. import T

from .common import ExponentialFamily
from .iw import IW
from .mn import MN

__all__ = ["MatrixNormalInverseWishart", "MNIW"]

class MatrixNormalInverseWishart(ExponentialFamily):

    def get_param_dim(self):
        return [2, 2, 2, 0]

    def expected_value(self):
        S, M0, V, nu = self.get_parameters('regular')
        return IW([S, nu]).expected_value(), M0

    def sample(self, num_samples=1):
        S, M0, V, nu = self.get_parameters('regular')
        sigma_sample = IW([S, nu]).sample(num_samples=num_samples)
        S_dim = list(S.get_shape()[-2:])
        V_dim = list(V.get_shape()[-2:])
        tiled_M0 = T.tile(M0[None], T.concatenate([[num_samples], T.cast(T.ones(T.rank(M0)), T.int32)]))
        tiled_V = T.tile(V[None], T.concatenate([[num_samples], T.cast(T.ones(T.rank(V)), T.int32)]))
        sigma_sample.set_shape(list(sigma_sample.get_shape()[:-2]) + S_dim)
        tiled_V.set_shape(list(tiled_V.get_shape()[:-2]) + V_dim)
        M_sample = MN([tiled_M0, sigma_sample, tiled_V]).sample()[0]
        M_sample = T.reshape(M_sample, T.concatenate([[num_samples], T.shape(M0)]))
        return sigma_sample, M_sample

    @classmethod
    def regular_to_natural(cls, regular_parameters):
        S, M0, V, nu = regular_parameters
        V_inv = T.matrix_inverse(V)
        M0V_1 = T.matmul(V_inv, T.matrix_transpose(M0))
        shape = T.cast(T.shape(M0), T.dtype(S))
        a, b = shape[-1], shape[-2]
        return [
            S + T.matmul(M0, M0V_1),
            T.matrix_transpose(M0V_1),
            V_inv,
            nu + a + b + 1
        ]

    @classmethod
    def natural_to_regular(cls, natural_parameters):
        A, B, V_inv, nu = natural_parameters
        V = T.matrix_inverse(V_inv)
        M0 = T.matmul(B, V)
        shape = T.cast(T.shape(M0), T.dtype(A))
        a, b = shape[-1], shape[-2]
        S = A - T.matmul(B, T.matrix_transpose(M0))
        return([
            S,
            M0,
            V,
            nu - a - b - 1
        ])

    def log_likelihood(self, x):
        pass

    def log_z(self):
        S, M0, V, nu = self.natural_to_regular(self.get_parameters('natural'))
        shape = T.cast(T.shape(M0), T.dtype(S))
        d, s = shape[-2], shape[-1]
        return (nu / 2. * (T.to_float(d) * T.log(2.) - T.logdet(S))
                + T.multigammaln(nu / 2., d)
                + T.to_float(s) / 2. * T.logdet(V))

    def log_h(self, x):
        raise NotImplementedError

    def sufficient_statistics(self, x):
        sigma, A = x
        sigma_inv_A = T.matrix_solve(sigma, A)
        return [
            -0.5 * T.matrix_inverse(sigma),
            sigma_inv_A,
            -0.5 * T.matmul(T.matrix_transpose(A), sigma_inv_A),
            -0.5 * T.logdet(sigma)
        ]

    def expected_sufficient_statistics(self):
        return T.grad(self.log_z(), self.get_parameters('natural'))
        S, M0, V, nu = self.get_parameters('regular')
        S_inv = T.matrix_inverse(S)
        S_inv_M0 = T.matrix_solve(S, M0)
        s = T.shape(V)[-1]
        d = T.shape(S)[-1]
        return [
            -nu[..., None, None] / 2. * S_inv,
            nu[..., None, None] * S_inv_M0,
            -nu[..., None, None] / 2. * T.matmul(T.matrix_transpose(M0), S_inv_M0) - T.cast(s, T.dtype(V)) / 2. * V,
            0.5 * (T.cast(d, T.dtype(S)) * T.cast(T.log(2.), T.dtype(S)) - T.logdet(S))
                + T.sum(T.digamma((nu[..., None] - T.cast(T.range(d)[None], T.dtype(S)))/2.), [-1, -2])
        ]

MNIW = MatrixNormalInverseWishart
