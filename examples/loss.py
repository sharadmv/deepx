import numpy as np
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    mlp = Vector(784) >> Repeat(Tanh(200) >> Dropout(0.5), 2) >> Softmax(10)
    loss = CrossEntropy()
    rmsprop = RMSProp(mlp, loss)
