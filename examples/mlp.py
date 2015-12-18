from deepx.nn import *

if __name__ == "__main__":
    mlp = Vector('X', 784) >> Tanh(200) >> Tanh(200) >> Softmax(10)
