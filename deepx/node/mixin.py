from .. import backend as T

class Mixin(object):

    name = None
    priority = 0

    def setup(self, model):
        self.model = model
        self.arch = model.arch
        self.func = None
        assert self.arch.is_initialized()

    def get_function(self):
        if self.func is not None:
            return self.func

        inputs = self.get_inputs()

        self.inputs = []

        for input in inputs:
            if input not in self.inputs:
                self.inputs.append(input)

        self.result = self.get_result()

        self.func = self.create_function()
        return self.func

    def get_inputs(self):
        input = self.arch.get_input()
        if not isinstance(input, list):
            input = [input]
        return [i.get_data() for i in input]

    def get_activation(self):
        input = self.arch.get_input()
        return self.arch.forward(input)

    def get_result(self):
        return self.get_activation().get_data()

    def get_updates(self):
        return []

    def create_function(self, **kwargs):
        return T.function(self.inputs, [self.result],
                          updates=self.get_updates(),
                               **kwargs)
