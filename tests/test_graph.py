import numpy as np
import unittest
from deepx.nn import Vector, Linear
from deepx.node import ShapeException

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
        net1 = Vector(10) >> Linear(10)

        np.random.seed(0)
        net2 = Vector(10).chain(Linear(20))

        self.assertNotEqual(net1, net2)

    def test_shape_inference(self):
        part1 = Linear(100)
        part2 = Linear(20)

        self.assertEqual(part1.shape, (None, 100))
        self.assertEqual(part2.shape, (None, 20))

        part1.chain(part2)

        self.assertEqual(part1.shape, (None, 100))
        self.assertEqual(part2.shape, (100, 20))

    def test_shape_inference2(self):
        part1 = Linear(10, 100)
        part2 = Linear(100, 20)

        self.assertEqual(part1.shape, (10, 100))
        self.assertEqual(part2.shape, (100, 20))

        part1.chain(part2)

        self.assertEqual(part1.shape, (10, 100))
        self.assertEqual(part2.shape, (100, 20))

    def test_bad_shape(self):
        part1 = Linear(10, 100)
        part2 = Linear(90, 20)

        with self.assertRaises(ShapeException):
            part1.chain(part2)

    def test_bad_shape2(self):
        part1 = Linear(100)
        part2 = Linear(20)

        self.assertEqual(part1.shape, (None, 100))
        self.assertEqual(part2.shape, (None, 20))

        part1.chain(part2)

        self.assertEqual(part1.shape, (None, 100))
        self.assertEqual(part2.shape, (100, 20))

        part3 = Linear(100)
        part2.chain(part3)

        part4 = Linear(90)
        with self.assertRaises(ShapeException):
            part4.chain(part3)

    def test_shape_elementwise(self):
        part1 = Linear()
        part2 = Linear(20)

        self.assertEqual(part1.shape, (None, None))
        self.assertEqual(part2.shape, (None, 20))

        part2.chain(part1)

        self.assertEqual(part1.shape, (20, 20))
        self.assertEqual(part2.shape, (None, 20))

    def test_shape_elementwise2(self):
        part1 = Linear(20)
        part2 = Linear()
        part3 = Linear(30, 40)

        self.assertEqual(part1.shape, (None, 20))
        self.assertEqual(part2.shape, (None, None))
        self.assertEqual(part3.shape, (30, 40))

        part4 = part1.chain(part2)

        self.assertEqual(part1.shape, (None, 20))
        self.assertEqual(part2.shape, (20, 20))
        self.assertEqual(part4.shape, (None, 20))

        with self.assertRaises(ShapeException):
            part4.chain(part3)
