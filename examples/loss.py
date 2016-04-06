import numpy as np
from deepx.nn import *
from deepx.rnn import *
from deepx.loss import *
from deepx.optimize import *

if __name__ == "__main__":
    X = Sequence(Vector(784))
    mlp = X >> LSTM(20) >> Softmax(2)
    loss = AdversarialLoss(mlp >> CrossEntropy(), X)
    adam = Adam(loss)
