from ..core import HOF

Residual = HOF(lambda a, b: (a >> b) + a)
