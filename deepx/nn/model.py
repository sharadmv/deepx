import logging
import theano
import numpy as np
from helper import floatX
from theanify import Theanifiable

class ParameterModel(Theanifiable):
    """
    A ParameterModel is an object containing compilable methods and parameters which are updated by these methods given sume input.
    parameters  A dictionary of parameters
    """
    def __init__(self, name):
        super(ParameterModel, self).__init__()
        self.name = name
        self.parameters = {}

    def initialize_weights(self, shape):
        return floatX(np.random.randn(*shape) * 0.01)

    def init_parameter(self, name, value):
        logging.debug("Creating parameter %s-%s" % (self.name, name))
        assert name not in self.parameters, "Cannot re-initialize theano shared variable, use set_parameter_value"
        self.parameters[name] = theano.shared(value, name='%s-%s' % (self.name, name))

    def get_parameter(self, name):
        return self.parameters[name]

    def set_parameter_value(self, name, value):
        return self.parameters[name].set_value(value)

    def get_parameter_value(self, name):
        return self.parameters[name].get_value()

    def get_parameters(self):
        return self.parameters.values()
