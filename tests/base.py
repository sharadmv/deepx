import unittest

class BaseTest(unittest.TestCase):

    def create_function(self, net):
        self.assertTrue(net.is_configured())
        return net.predict
