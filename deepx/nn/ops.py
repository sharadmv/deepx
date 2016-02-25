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
