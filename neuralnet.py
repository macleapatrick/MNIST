import numpy as np
import matplotlib.pyplot as plt

class Layer():
    def __init__(self, input_size, nodes, activation=None, layerid=0, addBias=True):
        self.addBias = addBias
        
        if self.addBias:
            self.inputs = input_size + 1
        else:
            self.inputs = input_size
            
        self.nodes = nodes
        self.weights = self._xavier_init(self.inputs, nodes)
        self.activationFunction = activation
        self.layerid = layerid
        
    def _xavier_init(self, input_size, output_size):
        return np.random.randn(input_size, output_size) * np.sqrt(2.0 / (input_size + output_size))

    def forwards(self, inputs):
        if self.addBias:
            if len(inputs.shape) == 1:
                inputs_wbias = np.insert(inputs, 0, 1, axis=0)
            elif len(inputs.shape) == 2:
                inputs_wbias = np.insert(inputs, 0, 1, axis=1)
            else:
                #only 1d or 2d
                raise(ValueError(f'{inputs.shape} can only 1d or 2d'))
        else:
            inputs_wbias = inputs

        if not self._canMultiply(inputs_wbias, self.weights):
            raise(ValueError(f'{inputs_wbias.shape} can not be broadcast with {self.weights.shape}'))

        return np.matmul(inputs_wbias, self.weights)
    
    def backwards(self, backprop_errors):
        if not self._canMultiply(backprop_errors, self.weights[1:].T):
            raise(ValueError(f'{backprop_errors.shape} can not be broadcast with {self.weights[1:].T.shape}'))
        
        #calculate backwards without bias weights
        return np.matmul(backprop_errors, self.weights[1:].T)
    
    def activation(self, activations):
        if self.activationFunction:
            return self.activationFunction.calc(activations)
        
    def _canMultiply(self, m1, m2):
        sh1 = np.shape(m1)
        sh2 = np.shape(m2)

        if len(sh1) == 1 and sh1[0] != sh2[0]:
            return False
        elif len(sh1) == 2 and sh1[1] != sh2[0]:
            return False
        else:
            return True

class BackpropStructure():
    def __init__(self):
        self.preActivations = []
        self.postActivations = []
        self.delta = []
        self.loss = []
        self.wGradient = []
        
    def clear(self):
        self.preActivations = []
        self.postActivations = []
        self.delta = []
        self.loss = []
        self.wGradient = []

class Network():
    def __init__(self, structure, loss):
        self.llayers = len(structure)
        self.layers = []
        self.batchSize = 0
        self.training = False
        self.lf = loss
        
        self.backprop = BackpropStructure()
        
        for layer in range(self.llayers):
            self.layers.append(Layer(input_size=structure[layer]['inputs'], 
                                     nodes=structure[layer]['nodes'],
                                     activation=structure[layer]['activation'],
                                     layerid=layer,
                                     addBias=True))
            
    def forward(self, inputs, targets=None):
        
        self.backprop.clear()
        self.batchSize = len(inputs)
        
        x = inputs
        
        if self.training:
            self.backprop.postActivations.append(x)
            self.backprop.preActivations.append(x)
            
        for layer in self.layers:
            x = layer.forwards(x)
            x_act = layer.activation(x)
            
            if self.training:
                self.backprop.preActivations.append(x)
                self.backprop.postActivations.append(x_act)
            
            #set input for next layer to post activations
            x = x_act
                
        outputs = x_act
            
        if self.training:
            self.backprop.loss = self.calcLoss(layer.activationFunction.reverse(outputs), layer.activationFunction.reverse(targets))
            self.backprop.delta.append(layer.activationFunction.reverse(outputs) - layer.activationFunction.reverse(targets))

        return outputs
    
    def backward(self):
            
        #backpropogate deltas
        for layer in range(self.llayers-1,0,-1):
            #calculate forward prop activations through the derivative of the layers activation function
            activations = self.layers[layer].activationFunction.derivative(self.backprop.preActivations[layer])
            
            #backprop deltas to previous layer
            deltaNextLayer = self.backprop.delta[0]
            deltaThisLayer = self.layers[layer].backwards(deltaNextLayer) * activations
            
            self.backprop.delta.insert(0, deltaThisLayer)
                
        #calculate gradient for each weight
        for layer in range(self.llayers):
            gradient = []
            activations = self.backprop.postActivations[layer]
            
            #add bias activations back for gradient calculation
            activations_wbias = np.insert(activations, 0, 1, axis=1)
                
            for batch_sample in range(self.batchSize):
                
                #Calculate gradient for every weight in this layer with every data sample
                activations_reshaped = activations_wbias[batch_sample].reshape((len(activations_wbias[batch_sample]),1))
                delta_reshaped = self.backprop.delta[layer][batch_sample].reshape((1,len(self.backprop.delta[layer][batch_sample])))
                    
                gradient.append(np.matmul(activations_reshaped, delta_reshaped))
            
            self.backprop.wGradient.insert(layer, sum(gradient))
                
        return self.backprop.wGradient
    
    def sgd(self, gradient, learning_rate=0.001):
        for i, layer in enumerate(self.layers):
            if layer.weights.shape != gradient[i].shape:
                raise(ValueError(f'{layer.weights.shape} not the same size as {gradient[i].shape}'))
            layer.weights = layer.weights - learning_rate*gradient[i]
        
    def calcLoss(self, y, t):
        # weight decay needs to be reshaped to fit loss matrix shape
        return self.lf.loss(y, t) #+ self.weightDecayLoss()

    def train(self):
        self.training = True
        self.loss = np.empty(0)
        return 1

    def evaluate(self):
        self.training = False
        self.loss = np.empty(0)
        return 1
    
    def weightDecayLoss(self, l=2):
        return 0


class RegressionLoss():
    def __init__(self):
        pass
    
    def loss(self, y, t):
        if y.shape == t.shape:
            #squared loss
            return 1/2 * pow(y-t,2)
        else:
            raise(f'loss cannot be calculated across {y.shape} and {t.shape}')
    
    
class ClassificationLoss():
    def __init__(self):
        pass
    
    def loss(self, y, t):
        if y.shape == t.shape:
            #cross entropy loss
            return -(t*np.log(y) + (1-t)*np.log(1-y))
        else:
            raise(f'loss cannot be calculated across {y.shape} and {t.shape}')
            

class ReLU():
    def __init(self):
        pass
    
    def calc(self, x):
        return np.maximum(0, x)
        
    def derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def reverse(self, x):
        return x
    
    
class Linear():
    def __init(self):
        pass
    
    def calc(self, x):
        return x
        
    def derivative(self, x):
        return np.ones(x.shape)
    
    def reverse(self, x):
        return x
    
    
class Sigmoid():
    def __init__(self):
        pass
    
    def calc(self, x):
        return 1 / (1 + np.exp(-x))
        
    def derivative(self, x):
        sigmoid_x = self.calc(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def reverse(self, x):
        # Avoid issues with log and division by zero
        x = np.clip(x, 1e-7, 1 - 1e-7)
        return np.log(x / (1 - x))
    
"""
class TanH():
    def __init(self):
        pass
    
    def calc(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        
    def derivative(self, x):
        return 1 - pow((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),2)
"""
        

network = [{'inputs':1,  'nodes':10, 'activation': Sigmoid()},
           {'inputs':10, 'nodes':10, 'activation': Sigmoid()},
           {'inputs':10, 'nodes':1,   'activation': Linear()}]


test_nn = Network(structure=network, loss=RegressionLoss())

inputs_train0 = np.random.randint(-5,5,(100,1)) + np.random.rand(100,1)
inputs_train1 = inputs_train0**2
inputs_train = np.hstack((inputs_train0, inputs_train1))
targets_train = np.sin(inputs_train0) + 2 * np.random.rand(100,1) - 1

inputs_validate0 = np.random.randint(-5,5,(100,1)) + np.random.rand(100,1)
inputs_validate1 = inputs_validate0**2
inputs_validate = np.hstack((inputs_validate0, inputs_validate1))
targets_validate = np.sin(inputs_validate0) + 2 * np.random.rand(100,1) - 1

test_nn.train()

for i in range(0,1000):
    print(f'epoch: {i}')
    test_nn.train()
    out = test_nn.forward(inputs_train0, targets_train)
    print('training loss:' + str(sum(test_nn.backprop.loss)))
    grad = test_nn.backward()
    test_nn.sgd(grad)
    
    out = test_nn.forward(inputs_validate0, targets_validate)
    print('validation loss:' + str(sum(test_nn.backprop.loss)))


test_nn.evaluate()
out_plot = test_nn.forward(inputs_validate0)
plt.scatter(inputs_train0, targets_train, color='y')
plt.scatter(inputs_validate0, targets_validate, color='b')
plt.scatter(inputs_validate0, out_plot, color='r')

        
