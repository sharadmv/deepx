import numpy as np
import unittest
from deepx.nn import Vector, Tanh, Linear

class TestGraph(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_graph_basic(self):
        np.random.seed(0)
        net1 = Vector(10) >> Linear(10)

        np.random.seed(0)
        net2 = Vector(10).chain(Linear(10))

        self.assertEqual(net1, net2)

    def test_graph_basic2(self):
        np.random.seed(0)
        net1 = Vector(10) >> Tanh(10)

        np.random.seed(0)
        net2 = Vector(10).chain(Tanh(20))

        self.assertNotEqual(net1, net2)

    def test_shape_inference(self):
        part1 = Tanh(100)
        part2 = Tanh(20)

        self.assertEqual(part1.shape, (None, 100))
        self.assertEqual(part2.shape, (None, 20))

        part1.chain(part2)

        self.assertEqual(part1.shape, (None, 100))
        self.assertEqual(part2.shape, (100, 20))

    def test_shape_inference2(self):
        part1 = Tanh(10, 100)
        part2 = Tanh(100, 20)

        self.assertEqual(part1.shape, (10, 100))
        self.assertEqual(part2.shape, (100, 20))

        part1.chain(part2)

        self.assertEqual(part1.shape, (10, 100))
        self.assertEqual(part2.shape, (100, 20))

    def test_bad_shape(self):
        part1 = Tanh(10, 100)
        part2 = Tanh(90, 20)

        self.assertRaises(lambda: part1.chain(part2))

    def test_bad_shape2(self):
        part1 = Tanh(100)
        part2 = Tanh(20)

        self.assertEqual(part1.shape, (None, 100))
        self.assertEqual(part2.shape, (None, 20))

        part1.chain(part2)

        self.assertEqual(part1.shape, (None, 100))
        self.assertEqual(part2.shape, (100, 20))

        part3 = Tanh(100)
        part3.chain(part2)

        part4 = Tanh(90)
        self.assertRaises(lambda: part3.chain(part2))

    def test_shape_elementwise(self):
        part1 = Tanh()
        part2 = Tanh(20)

        self.assertEqual(part1.shape, (None, None))
        self.assertEqual(part2.shape, (None, 20))

        part2.chain(part1)

        self.assertEqual(part1.shape, (20, 20))
        self.assertEqual(part2.shape, (None, 20))
