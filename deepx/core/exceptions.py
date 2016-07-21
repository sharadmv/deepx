class ShapeOutError(Exception):

    def __init__(self, node, shape_out):
        message = "Shape mismatch for %s. Expected out: %s. Inferred out: %s" % (
            node,
            str(node.get_shapes_out()),
            str(shape_out),
        )
        super(ShapeOutError, self).__init__(message)

class ShapeInError(Exception):

    def __init__(self, node, shape_in):
        message = "Shape mismatch for %s. Expected in: %s. Inferred in: %s" % (
            node,
            str(node.get_shapes_in()),
            str(shape_in),
        )
        super(ShapeInError, self).__init__(message)
