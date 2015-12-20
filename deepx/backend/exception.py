class BackendException(Exception):

    def __init__(self, method):
        self.method = method
