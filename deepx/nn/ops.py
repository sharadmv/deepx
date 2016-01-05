def Repeat(node,  n_repeats):
    new_node = node.copy()
    for i in range(n_repeats - 1):
        new_node = new_node.chain(node.copy())
    return new_node

MLP = Repeat
