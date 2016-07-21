from base import BaseTest

import numpy as np
from deepx.nn import Vector, Linear, Tanh, Relu, Sigmoid, Softmax, Elu
import deepx.backend as T

class TestLinear(BaseTest):

    def run_simple_full(self, type, activation, in_size=2, out_size=3, batch_size=5, **kwargs):
        np.random.seed(0)

        layer = type(in_size, out_size, **kwargs)
        input = Vector(in_size)

        net_func = self.create_function(input >> layer)

        B = batch_size
        X = np.random.normal(size=(B, in_size))

        result = net_func(X)

        state = layer.get_state()
        W, b = np.array(state['W']), np.array(state['b'])
        result2 = activation(np.dot(X, W) + b)

        np.testing.assert_almost_equal(result, result2, decimal=5)

    def run_simple_elementwise(self, type, activation, in_size=2, out_size=3, batch_size=5, **kwargs):
        np.random.seed(0)

        layer = type(**kwargs)
        input = Vector(in_size)

        net_func = self.create_function(input >> layer)

        B = batch_size
        X = np.random.normal(size=(B, in_size))

        result = net_func(X)
        result2 = activation(X)

        np.testing.assert_almost_equal(result, result2, decimal=5)

    def test_full(self):
        self.run_simple_full(Linear, lambda x: x)
        self.run_simple_full(Linear, lambda x: x, 5, 10)

    def test_tanh(self):
        self.run_simple_full(Tanh, np.tanh)
        self.run_simple_full(Tanh, np.tanh, 5, 10)

    def test_relu(self):
        self.run_simple_full(Relu, lambda x: x * (x > 0))
        self.run_simple_full(Relu, lambda x: x * (x > 0), 5, 10)

    def test_elu(self):
        def make_elu(alpha=1.0):
            def elu(x):
                return x * (x > 0) + alpha * (np.exp(x) - 1) * (x <= 0)
            return elu
        self.run_simple_full(Elu, make_elu())
        self.run_simple_full(Elu, make_elu(), 5, 10)
        self.run_simple_full(Elu, make_elu(5.0), 5, 10, alpha=5.0)
        self.run_simple_full(Elu, make_elu(0.1), 5, 10, alpha=0.1)

    def test_sigmoid(self):
        self.run_simple_full(Sigmoid, lambda x: 1.0 / (1 + np.exp(-x)))
        self.run_simple_full(Sigmoid, lambda x: 1.0 / (1 + np.exp(-x)), 5, 10)

    def test_full_element(self):
        with self.assertRaises(Exception):
            self.run_simple_elementwise(Linear, lambda x: x)

    def test_tanh_element(self):
        self.run_simple_elementwise(Tanh, np.tanh)
        self.run_simple_elementwise(Tanh, np.tanh, 5, 10)

    def test_relu_element(self):
        self.run_simple_elementwise(Relu, lambda x: x * (x > 0))
        self.run_simple_elementwise(Relu, lambda x: x * (x > 0), 5, 10)

    def test_elu_element(self):
        def make_elu(alpha=1.0):
            def elu(x):
                return x * (x > 0) + alpha * (np.exp(x) - 1) * (x <= 0)
            return elu
        self.run_simple_elementwise(Elu, make_elu())
        self.run_simple_elementwise(Elu, make_elu(), 5, 10)
        self.run_simple_elementwise(Elu, make_elu(5.0), 5, 10, alpha=5.0)
        self.run_simple_elementwise(Elu, make_elu(0.1), 5, 10, alpha=0.1)

    def test_sigmoid_element(self):
        self.run_simple_elementwise(Sigmoid, lambda x: 1.0 / (1 + np.exp(-x)))
        self.run_simple_elementwise(Sigmoid, lambda x: 1.0 / (1 + np.exp(-x)), 5, 10)
