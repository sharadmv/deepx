import numpy as np
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    X = Vector(784)
    mlp = X >> Repeat(Tanh(200), 2) >> Softmax(10)
    loss = AdversarialLoss(mlp >> CrossEntropy(), X)
    adam = Adam(loss)
