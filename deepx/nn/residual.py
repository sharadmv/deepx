def Residual(node):
    def ResidualNode(x):
        return x + node(x)
    ResidualNode.__name__ = 'Residual(%s)' % str(node)
    return ResidualNode
