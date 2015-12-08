import theano

class Mixin(object):

    name = None

    def setup(self, model):
        self.arch = model.arch
        self.inputs = self.get_inputs()
        self.result = self.get_result()

        self.func = self.create_function()

    def get_result(self):
        raise NotImplementedError

    def get_inputs(self):
        raise NotImplementedError

    def create_function(self):
        return theano.function(self.inputs, self.result)

class predict(Mixin):

    name = 'predict'

    def get_inputs(self):
        return [i.get_data() for i in self.arch.get_inputs()]

    def get_result(self):
        return self.arch.get_activation().get_data()
