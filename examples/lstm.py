import theano.tensor as T
import numpy as np

from deepx.optimize import RMSProp
from deepx.layer import *

if __name__ == "__main__":
    data_dim = 1

    model = Softmax(784, 10) | output
    rmsprop = RMSProp(model)
