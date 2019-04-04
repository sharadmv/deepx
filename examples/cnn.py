from deepx.backend import T
from deepx import nn

net = (nn.Reshape([28, 28, 1])
            >> nn.Conv([2, 2, 64])
            >> nn.Conv([2, 2, 32])
            >> nn.Conv([2, 2, 16])
            >> nn.Flatten() >> nn.Relu(200) >> nn.Relu(200) >> nn.Softmax(10))

result = net(T.random_normal([10, 784]))
