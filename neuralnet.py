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
        self.inputSize = 0
        self.targetSize = 0
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
        self.batchSize = inputs.shape[0]
        
        try:
            self.inputSize = inputs.shape[1]
        except IndexError:
            self.inputSize = 1
        
        x = inputs
        
        if self.training:
            self.backprop.postActivations.append(x)
            self.backprop.preActivations.append(x)
            
            try:
                self.targetSize = targets.shape[1]
            except IndexError:
                self.targetSize = 1
            
            if targets.shape[0] != self.batchSize:
                raise(TypeError, f'number of input batches: {self.batchSize} does not match number of output batches: {targets.shape[0]}')
            
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
            outputs = outputs.reshape(self.batchSize, self.targetSize)
            targets = targets.reshape(self.batchSize, self.targetSize)

            self.backprop.loss = self.calcLoss(outputs, targets)
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
    
    def sgd(self, gradient, lr=0.001, lb=0):
        for i, layer in enumerate(self.layers):
            if layer.weights.shape != gradient[i].shape:
                raise(ValueError(f'{layer.weights.shape} not the same size as {gradient[i].shape}'))
            layer.weights = (1-lr*lb)*layer.weights - lr*gradient[i]
        
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
    
    
class LeakyReLU():
    def __init(self):
        pass
    
    def calc(self, x):
        return np.maximum(0.1*x, x)
        
    def derivative(self, x):
        return np.where(x > 0, 1, 0.1)
    
    def reverse(self, x):
        raise(ValueError, "NOT IMPLIMENTED YET")
        return 0
    
    
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
        x = np.clip(x, 1e-2, 1 - 1e-2)
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

"""
class Softmax():
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
        

network = [{'inputs':2,  'nodes':100, 'activation': ReLU()},
           {'inputs':100, 'nodes':100, 'activation': ReLU()},
           {'inputs':100, 'nodes':100, 'activation': ReLU()},
           {'inputs':100, 'nodes':1,  'activation': Sigmoid()}]


test_nn = Network(structure=network, loss=ClassificationLoss())

inputs_train00 = np.ones((100,1)) + np.random.randn(100,1)
inputs_train01 = np.ones((100,1)) + np.random.randn(100,1)
targets0 = np.zeros((100,1))

inputs_train10 = 2*np.ones((100,1)) + np.random.randn(100,1)
inputs_train11 = 2*np.ones((100,1)) + np.random.randn(100,1)
target1 = np.ones((100,1))

X = np.zeros((200,3))
X[:,0] = np.concatenate([inputs_train00, inputs_train10]).T
X[:,1] = np.concatenate([inputs_train01, inputs_train11]).T
X[:,2] = np.concatenate([targets0, target1]).T

np.random.shuffle(X)

test_nn.train()


for i in range(0,1000):
    print(f'epoch: {i}')
    test_nn.train()
    out = test_nn.forward(X[:,0:2], X[:,2])
    print('training loss:' + str(sum(test_nn.backprop.loss)))
    grad = test_nn.backward()
    test_nn.sgd(grad, lr=0.001, lb=0)
    
    
test_nn.evaluate()

x11 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 10)
x22 = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 10)

grid = np.meshgrid(x11,x22)
positions = np.vstack([grid[0].ravel(), grid[1].ravel()]).T
        
db_inputs = np.asarray(positions)
db_outs = test_nn.forward(db_inputs)

x0 = X[np.where(X[:,2]==0),:]
x1 = X[np.where(X[:,2]==1),:]
z = db_outs.reshape((10,10))

plt.scatter(x0[0][:,0], x0[0][:,1], color='y')
plt.scatter(x1[0][:,0], x1[0][:,1], color='b')
plt.contour(grid[0], grid[1], z, levels=[0.25, 0.5, 0.75])

        
