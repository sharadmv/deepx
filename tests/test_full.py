from base import BaseTest

import numpy as np
from deepx.nn import Vector, Full, Tanh, Relu, Sigmoid, Softmax, Elu
import deepx.backend as T

class TestFull(BaseTest):

    def run_simple_full(self, type, activation, in_size=2, out_size=3, batch_size=5):
        np.random.seed(0)

        layer = type(in_size, out_size)
        input = Vector(in_size)

        state = layer.get_state()
        W, b = np.array(state['W']), np.array(state['b'])

        net_func = self.create_function(input >> layer)

        B = batch_size
        X = np.random.normal(size=(B, in_size))

        result = net_func(X)
        result2 = activation(np.dot(X, W) + b)

        np.testing.assert_almost_equal(result, result2)

    def run_simple_elementwise(self, type, activation, in_size=2, out_size=3, batch_size=5):
        np.random.seed(0)

        layer = type()
        input = Vector(in_size)

        net_func = self.create_function(input >> layer)

        B = batch_size
        X = np.random.normal(size=(B, in_size))

        result = net_func(X)
        result2 = activation(X)

        np.testing.assert_almost_equal(result, result2)

    def test_full(self):
        self.run_simple_full(Full, lambda x: x)
        self.run_simple_full(Full, lambda x: x, 5, 10)

    def test_tanh(self):
        self.run_simple_full(Tanh, np.tanh)
        self.run_simple_full(Tanh, np.tanh, 5, 10)

    def test_relu(self):
        self.run_simple_full(Relu, lambda x: x * (x > 0))
        self.run_simple_full(Relu, lambda x: x * (x > 0), 5, 10)

    def test_elu(self):
        def elu(x):
            return x * (x > 0) + (np.exp(x) - 1) * (x <= 0)
        self.run_simple_full(Elu, elu)
        self.run_simple_full(Elu, elu, 5, 10)

    def test_sigmoid(self):
        self.run_simple_full(Sigmoid, lambda x: 1.0 / (1 + np.exp(-x)))
        self.run_simple_full(Sigmoid, lambda x: 1.0 / (1 + np.exp(-x)), 5, 10)

    def test_full_element(self):
        with self.assertRaises(Exception):
            self.run_simple_elementwise(Full, lambda x: x)

    def test_tanh_element(self):
        self.run_simple_elementwise(Tanh, np.tanh)
        self.run_simple_elementwise(Tanh, np.tanh, 5, 10)

    def test_relu_element(self):
        self.run_simple_elementwise(Relu, lambda x: x * (x > 0))
        self.run_simple_elementwise(Relu, lambda x: x * (x > 0), 5, 10)

    def test_elu_element(self):
        def elu(x):
            return x * (x > 0) + (np.exp(x) - 1) * (x <= 0)
        self.run_simple_elementwise(Elu, elu)
        self.run_simple_elementwise(Elu, elu, 5, 10)

    def test_sigmoid_element(self):
        self.run_simple_elementwise(Sigmoid, lambda x: 1.0 / (1 + np.exp(-x)))
        self.run_simple_elementwise(Sigmoid, lambda x: 1.0 / (1 + np.exp(-x)), 5, 10)
