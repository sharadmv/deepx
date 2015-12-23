
class ShapeException(Exception):

    def __init__(self, node, shape_in):
        message = "Shape mismatch for %s. Expected in: %s. Inferred in: %s" % (
            node,
            str(node.get_shape_in()),
            str(shape_in),
        )
        super(ShapeException, self).__init__(message)
