import logging
logging.basicConfig(level=logging.INFO)
from deepx.nn import *
from deepx.optimize import *

if __name__ == "__main__":
    mlp = Vector('X', 784) >> Relu(200) >> Tanh(200) >> Softmax(10) | (predict, cross_entropy, rmsprop)
