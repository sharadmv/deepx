import numpy as np
from base import BaseTest

from deepx.nn import Lambda, Scalar, Repeat

class TestOps(BaseTest):

    def run_net(self, net, input, output):
        np.random.seed(0)

        net_func = self.create_function(net)

        result = net_func(input)
        np.testing.assert_almost_equal(result, output, decimal=5)

    def test_repeat1(self):
        add_one = Lambda(lambda x: x + 1)

        self.run_net(Scalar() >> add_one,            np.array([1]), np.array([2]))
        self.run_net(Scalar() >> Repeat(add_one, 2), np.array([1]), np.array([3]))
        self.run_net(Scalar() >> Repeat(add_one, 10),np.array([1]), np.array([11]))

    def test_repeat2(self):
        double = Lambda(lambda x: x * 2)

        self.run_net(Scalar() >> double,            np.array([1]), np.array([2]))
        self.run_net(Scalar() >> Repeat(double, 2), np.array([1]), np.array([4]))
        self.run_net(Scalar() >> Repeat(double, 4), np.array([1]), np.array([16]))

    def test_repeat3(self):
        double = Lambda(lambda x: x * 2)

        with self.assertRaises(AssertionError):
            self.run_net(Scalar() >> Repeat(double, 0), np.array([1]), np.array([0]))
        with self.assertRaises(AssertionError):
            self.run_net(Scalar() >> Repeat(double, -1), np.array([1]), np.array([0]))
