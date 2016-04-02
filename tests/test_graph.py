import numpy as np
from base import BaseTest

from deepx.nn import Vector, Linear, Freeze
from deepx.core import ShapeException

class TestGraph(BaseTest):

    def test_graph_basic(self):
        np.random.seed(0)
        v1 = Vector(10)
        v2 = Linear(10)

        self.assertEqual((v1 >> v2).left, v1.chain(v2).left)
        self.assertEqual((v1 >> v2).right, v1.chain(v2).right)

    def test_graph_parameters(self):
        np.random.seed(0)
        net1 = Vector(10) >> Linear(10)

        net2 = net1 >> Linear(10)

        self.assertEqual(net1.get_state(as_list=True), net2.left.get_state(as_list=True))

    def test_freeze(self):
        net1 = Vector(10) >> Linear(10)
        self.assertEqual(Freeze(net1).get_parameters(), [])

    def test_freeze_parameters(self):
        np.random.seed(0)
        net1 = Vector(10) >> Linear(10)

        self.assertEqual(net1.freeze().get_state(as_list=True), net1.get_state(as_list=True))

    def test_freeze_parameters2(self):
        np.random.seed(0)
        gan = (Vector(10) >> Linear(20)) >> (Linear(10) >> Linear(2))

        self.assertEqual(gan.left.freeze().get_state(as_list=True), gan.left.get_state(as_list=True))
        self.assertEqual(gan.right.freeze().get_state(as_list=True), gan.right.get_state(as_list=True))

        self.assertEqual(gan.right.freeze().get_parameters(), [])
        self.assertNotEqual(gan.right.get_parameters(), [])

        self.assertEqual((Vector(20) >> gan.right).freeze().get_parameters(), [])
        self.assertNotEqual((Vector(20) >> gan.right).get_parameters(), [])

        self.assertEqual((Vector(20) >> gan.right).freeze().get_state(as_list=True), (Vector(20) >> gan.right).get_state(as_list=True))

        self.assertEqual(gan.left.freeze().get_parameters(), [])
        self.assertNotEqual(gan.left.get_parameters(), [])

    def test_shape_inference(self):
        part1 = Linear(100)
        part2 = Linear(20)

        self.assertEqual(part1.get_shape(), (None, 100))
        self.assertEqual(part2.get_shape(), (None, 20))

        part1.chain(part2)

        self.assertEqual(part1.get_shape(), (None, 100))
        self.assertEqual(part2.get_shape(), (100, 20))

    def test_shape_inference2(self):
        part1 = Linear(10, 100)
        part2 = Linear(100, 20)

        self.assertEqual(part1.get_shape(), (10, 100))
        self.assertEqual(part2.get_shape(), (100, 20))

        part1.chain(part2)

        self.assertEqual(part1.get_shape(), (10, 100))
        self.assertEqual(part2.get_shape(), (100, 20))

    def test_bad_shape(self):
        part1 = Linear(10, 100)
        part2 = Linear(90, 20)

        with self.assertRaises(ShapeException):
            part1.chain(part2)

    def test_bad_shape2(self):
        part1 = Linear(100)
        part2 = Linear(20)

        self.assertEqual(part1.get_shape(), (None, 100))
        self.assertEqual(part2.get_shape(), (None, 20))

        part1.chain(part2)

        self.assertEqual(part1.get_shape(), (None, 100))
        self.assertEqual(part2.get_shape(), (100, 20))

        part3 = Linear(100)
        part2.chain(part3)

        part4 = Linear(90)
        with self.assertRaises(ShapeException):
            part4.chain(part3)

    def test_shape_elementwise(self):
        part1 = Linear()
        part2 = Linear(20)

        self.assertEqual(part1.get_shape(), (None, None))
        self.assertEqual(part2.get_shape(), (None, 20))

        part2.chain(part1)

        self.assertEqual(part1.get_shape(), (20, 20))
        self.assertEqual(part2.get_shape(), (None, 20))

    def test_shape_elementwise2(self):
        part1 = Linear(20)
        part2 = Linear()
        part3 = Linear(30, 40)

        self.assertEqual(part1.get_shape(), (None, 20))
        self.assertEqual(part2.get_shape(), (None, None))
        self.assertEqual(part3.get_shape(), (30, 40))

        part4 = part1.chain(part2)

        self.assertEqual(part1.get_shape(), (None, 20))
        self.assertEqual(part2.get_shape(), (20, 20))
        self.assertEqual(part4.get_shape(), (None, 20))

        with self.assertRaises(ShapeException):
            part4.chain(part3)
