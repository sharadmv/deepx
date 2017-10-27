# from ..core import HOF
from ..ops import Identity

def residualize(nodes):
    node, transform = nodes
    return node + transform

def Residual(node):
    dim_in, dim_out = node.get_dim_in(), node.get_dim_out()
    assert dim_in != None and dim_out != None
    if dim_in != dim_out:
        from ..nn import Linear
        return node + Linear(dim_in, dim_out)
    return node + Identity()

# Residual = HOF(lambda a, b: (a >> b) + a)
