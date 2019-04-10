from deepx.backend import T
from deepx.nn import Softmax
from deepx.rnn import LSTM

net = LSTM(2, 50) >> LSTM(50) >> Softmax(2)
result = net(T.ones([1, 10, 2]))
