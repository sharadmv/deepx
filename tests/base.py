import unittest

import deepx.backend as T

class BaseTest(unittest.TestCase):

    def create_function(self, net):
        self.assertTrue(net.is_initialized())
        input = net.get_input()
        output = net.forward(input)
        return T.function([input.get_data()], [output.get_data()])
