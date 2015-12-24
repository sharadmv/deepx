from base import BaseTest

import numpy as np
from deepx.nn import Vector, Flatten, Reshape, Matrix
import deepx.backend as T

class TestReshape(BaseTest):

    def create_function(self, net):
        self.assertTrue(net.is_initialized())
        input = net.get_input()
        output = net.forward(input)
        return T.function([input.get_data()], [output.get_data()])

    def test_reshape(self):
        input = Matrix((5, 5))

        layer = Reshape(25)

        net_func = self.create_function(input >> layer)

        np.random.seed(0)

        X = np.random.normal(size=(4, 5, 5))

        result = np.reshape(X, (4, 25))
        result2 = net_func(X)

        self.assertEqual(result.shape, result2.shape)
        np.testing.assert_almost_equal(result, result2)

    def test_reshape2(self):
        input = Vector(30)

        layer = Reshape((10, 3))

        net_func = self.create_function(input >> layer)

        np.random.seed(0)

        X = np.random.normal(size=(100, 30))

        result = np.reshape(X, (100, 10, 3))
        result2 = net_func(X)

        self.assertEqual(result.shape, result2.shape)
        np.testing.assert_almost_equal(result, result2)

    def test_flatten(self):
        input = Matrix((10, 3))

        reshape = Reshape(30)
        flatten = Flatten()

        net_func1 = self.create_function(input >> reshape)
        net_func2 = self.create_function(input >> flatten)

        np.random.seed(0)

        X = np.random.normal(size=(25, 10, 3))

        result = np.reshape(X, (25, 30))

        result1 = net_func1(X)
        result2 = net_func2(X)

        self.assertEqual(result.shape, result1.shape)
        self.assertEqual(result.shape, result2.shape)

        np.testing.assert_almost_equal(result, result1)
        np.testing.assert_almost_equal(result, result2)
