import numpy as np
from deepx.nn import Softmax, Tanh, predict, Vector
from deepx.rnn import LSTM
from deepx.optimize import rmsprop, cross_entropy

if __name__ == "__main__":
    lstm = (Vector('X', 100) >> LSTM(100) >> Softmax(10))()
    model = lstm | predict
