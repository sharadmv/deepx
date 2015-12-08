from deepx.rnn import LSTM
from deepx.layer import *
from deepx.optimize import rmsprop

if __name__ == "__main__":
    charrnn = Sequence('X') >> LSTM(10, 20) >> LSTM(20, 20) >> Softmax(20, 10) | (predict, cross_entropy, rmsprop)
