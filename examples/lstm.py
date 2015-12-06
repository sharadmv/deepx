import theano.tensor as T
import numpy as np

from deepx.layer import *

if __name__ == "__main__":
    X = Data(T.tensor3())
    lstm1 = LSTM(10, 20)
    lstm2 = LSTM(20, 20)
