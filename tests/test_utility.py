from base import BaseTest

import numpy as np
from deepx.nn import Vector, Lambda
import deepx.backend as T

class TestUtility(BaseTest):

    def run_lambda(self, func, np_func, shape_func=lambda x: x):
        np.random.seed(0)

        in_size = 10
        batch_size = 100

        layer = Lambda(func, shape_func=shape_func)
        input = Vector(in_size)

        net_func = self.create_function(input >> layer)

        B = batch_size
        X = np.random.normal(size=(B, in_size))

        result = net_func(X)
        result2 = np_func(X)

        np.testing.assert_almost_equal(result, result2, decimal=5)

    def test_lambda1(self):
        self.run_lambda(lambda x: x * 2, lambda x: x * 2)
        self.run_lambda(lambda x: T.pow(x, 2), lambda x: x ** 2)
        self.run_lambda(lambda x: abs(x), lambda x: abs(x))
