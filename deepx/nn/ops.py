def Repeat(node, n_repeats):
    assert n_repeats > 0, "Must repeat a positive number of times"
    new_node = node.copy()
    for i in range(n_repeats - 1):
        new_node = new_node.chain(node.copy())
    return new_node

MLP = Repeat

def Last(node):
    return node[-1]

def First(node):
    return node[0]

def Freeze(node):
    return node.freeze()

def Unfreeze(node):
    return node.unfreeze()

# class Norm(NestedNode):

    # def __init__(self, node, type='l2'):
        # super(Norm, self).__init__(node)
        # self.type = type

    # def _infer(self, shape_in):
        # assert isinstance(shape_in, int)
        # return ()

    # def forward(self, X, **kwargs):
        # X = self.node.forward(X, **kwargs)
        # assert isinstance(X.get_shape_out(), int)
        # if self.type == 'l2':
            # out = T.sum(T.pow(X.get_data(), 2), axis=-1)
        # else:
            # raise NotImplementedError(self.type)
        # return X.next(out, ())

