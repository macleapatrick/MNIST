import numpy

class Layer():
    def __init__(self, input_size, nodes, activation=None):
        self.inputs = input_size + 1
        self.nodes = nodes
        self.weights = numpy.random.rand(self.inputs, self.nodes) - 0.5
        self.activation = activation

    def forwards(self, inputs):
        inputs_wbias = numpy.insert(inputs, 0, 1, axis=1)

        if not self._canMultiply(inputs_wbias, self.weights):
            raise(ValueError(f'{inputs_wbias.shape} can not be broadcast with {self.weights.shape}'))

        return self._activation(numpy.matmul(inputs_wbias, self.weights)) 
    
    def backwards(self, backprop_errors):
        if len(backprop_errors) != self.nodes:
            raise(ValueError)
        
    def _activation(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + numpy.exp(-x))
        elif self.activation == 'relu':
            return x * (x >= 0)
        elif self.activation == 'tanh':
            return (numpy.exp(x) - numpy.exp(-x)) / (numpy.exp(x) + numpy.exp(-x))
        elif self.activation == 'softmax':
            return numpy.exp(x) / numpy.sum(numpy.exp(x),axis=1).reshape(-1,1)
        else:
            return x
        
    def _canMultiply(self, m1, m2):
        sh1 = numpy.shape(m1)
        sh2 = numpy.shape(m2)

        if len(sh1) == 1 and sh1[0] != sh2[0]:
            return False
        elif len(sh1) == 2 and sh1[1] != sh2[0]:
            return False
        else:
            return True


class Network():
    def __init__(self, structure):
        self.llayers = len(structure)
        self.layers = []
        self.training = False

        for layer in range(self.llayers):
            self.layers.append(Layer(input_size=structure[layer]['inputs'], 
                                     nodes=structure[layer]['nodes'],
                                     activation=structure[layer]['activation']))
            
    def forward(self, x):
        for layer in range(self.llayers):
            x = self.layers[layer].forwards(x)

        return x
    
    def backward(self, e):
        for layer in range(self.llayers, 0, -1):
            pass

    def train(self):
        self.training = True
        return 1

    def eval(self):
        self.training = False
        return 1


network = [{'inputs':100, 'nodes':100, 'activation':'relu'},
           {'inputs':100, 'nodes':100, 'activation':'relu'},
           {'inputs':100, 'nodes':1, 'activation':'softmax'}]


test_nn = Network(structure=network)

inputs = numpy.random.rand(100,100)

out = test_nn.forward(inputs)
        
print(out)

        
