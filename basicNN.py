import numpy as np
import copy
import matplotlib.pyplot as plt

class LinearLayer():
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
        
        #init layers
        for layer in range(self.llayers):
            self.layers.append(LinearLayer(input_size=structure[layer]['inputs'], 
                                           nodes=structure[layer]['nodes'],
                                           activation=structure[layer]['activation'],
                                           layerid=layer,
                                           addBias=True))
            
    def forward(self, inputs, targets=None):
        
        if not isinstance(inputs, np.ndarray):
            raise(TypeError, f'{type(inputs)} not type ndarray')
            
        if not isinstance(targets, np.ndarray) and self.training: 
            raise(TypeError, f'{type(targets)} not type ndarray')
        
        self.batchSize = inputs.shape[0]
        
        try:
            self.inputSize = inputs.shape[1]
        except IndexError:
            self.inputSize = 1
        
        x = inputs
        
        if self.training:
            self.backprop.clear()
            
            self.backprop.postActivations.append(x)
            
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
            
            if isinstance(layer.activationFunction, Softmax):
                #softmax requires preactivations for inversions
                self.backprop.delta.append(layer.activationFunction.outputLayerDelta(outputs, targets, self.backprop.preActivations[-1]))
            else:
                self.backprop.delta.append(layer.activationFunction.outputLayerDelta(outputs, targets))

        return outputs
    
    def backward(self):
            
        #backpropogate deltas
        for layer in range(self.llayers-1,0,-1):
            thisLayer = layer
            previousLayer= layer - 1
            
            #calculate forward prop activations through the derivative of the layers activation function
            activations = self.layers[previousLayer].activationFunction.derivative(self.backprop.preActivations[previousLayer])
            
            #backprop deltas to previous layer
            deltaNextLayer = self.backprop.delta[0]
            deltaThisLayer = self.layers[thisLayer].backwards(deltaNextLayer) * activations
            
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
    
    def sgd(self, gradient, lr=0.0001, lb=0):
        for i, layer in enumerate(self.layers):
            if layer.weights.shape != gradient[i].shape:
                raise(ValueError(f'{layer.weights.shape} not the same size as {gradient[i].shape}'))
                
            #calculate weight decay term
            weights_wo_bias = copy.deepcopy(layer.weights)
            weights_wo_bias[0,:] = 0
            weightDecay = lb*weights_wo_bias
            
            #gradient descent
            layer.weights = layer.weights - (lr*gradient[i] + lr*weightDecay)
        
    def calcLoss(self, y, t):
        # weight decay needs to be reshaped to fit loss matrix shape
        return self.lf.loss(y, t) #+ self.weightDecayLoss()

    def train(self):
        self.training = True
        self.backprop.clear()
        return 1

    def evaluate(self):
        self.training = False
        self.backprop.clear()
        return 1


class RegressionLoss():
    def __init__(self):
        pass
    
    def loss(self, y, t):
        if y.shape == t.shape:
            #mean squared loss
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
           
            
class ActivationFunction():
    # Container class for activation functions
    def __init__(self):
        pass
    
    def calc(self, x):
        xt = self.form(x)
        return self._calc(xt)
    
    def derivative(self, x):
        xt = self.form(x)
        return self._derivative(xt)
    
    def reverse(self, x, a=None):
        xt = self.form(x)
        if a is not None: at = self.form(a)
        return self._reverse(xt, at)
    
    def outputLayerDelta(self, y, t, a=None):
        yt = self.form(y)
        tt = self.form(t)
        if a is not None: a = self.form(a)
            
        return self._outputLayerDelta(yt, tt, a)
    
    def form(self, x):
        if not isinstance(x, (list, np.ndarray)):
            raise(TypeError(f'{type(x)} not data type list or ndarray'))
            
        if isinstance(x, list):
            x = np.array(x)
            
        return x


class ReLU(ActivationFunction):
    def __init(self):
        super().__init__()
    
    def _calc(self, x):
        return np.maximum(0, x)
        
    def _derivative(self, x):
        return np.where(x > 0, 1, 0)
    
    def _reverse(self, x, a=None):
        return x
    
    def _outputLayerDelta(self, outputs, targets, activations):
        return self._reverse(outputs) - self._reverse(targets)
    
    
class LeakyReLU(ActivationFunction):
    def __init(self):
        super().__init__()
    
    def _calc(self, x):
        return np.maximum(0.1*x, x)
        
    def _derivative(self, x):
        return np.where(x > 0, 1, 0.1)
    
    def _reverse(self, x, a=None):
        raise(ValueError, "NOT IMPLIMENTED YET")
        return 0
    
    def _outputLayerDelta(self, outputs, targets, activations):
        return self._reverse(outputs) - self._reverse(targets)
    
    
class Linear(ActivationFunction):
    def __init(self):
        super().__init__()
    
    def _calc(self, x):
        return x
        
    def _derivative(self, x):
        return np.ones(x.shape)
    
    def _reverse(self, x, a=None):
        return x
    
    def _outputLayerDelta(self, outputs, targets, activations):
        return self._reverse(outputs) - self._reverse(targets)
    
    
class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()
    
    def _calc(self, x):
        return 1 / (1 + np.exp(-x))
        
    def _derivative(self, x):
        sigmoid_x = self.calc(x)
        return sigmoid_x * (1 - sigmoid_x)
    
    def _reverse(self, x, a=None):
        # Avoid issues with log and division by zero
        x = np.clip(x, 1e-2, 1 - 1e-2)
        return np.log(x / (1 - x))
    
    def _outputLayerDelta(self, outputs, targets, activations):
        return self._reverse(outputs) - self._reverse(targets)
    
    
class TanH(ActivationFunction):
    def __init(self):
        super().__init__()
    
    def _calc(self, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        
    def _derivative(self, x):
        return 1 - pow((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)),2)
    
    def _reverse(self, x, a=None):
        raise(TypeError, 'Not completed')
        
    def _outputLayerDelta(self, outputs, targets, activations):
        return self._reverse(outputs) - self._reverse(targets)


class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__()
    
    def _calc(self, x):
        return np.transpose(np.exp(x).T / np.sum(np.exp(x), axis=len(x.shape)-1))
            
    def _derivative(self, x):
        # dont need derivative unless softmax is in a hidden layer
        raise(TypeError, 'Not completed')
    
    def _reverse(self, x, a): 
        x = np.clip(x, 1e-2, 1 - 1e-2)
        
        if len(a.shape) == 1: 
            samples = 1 
        else: 
            samples = len(a)
            
        return np.log(np.sum(np.exp(a), axis=len(a.shape)-1)).reshape(samples, 1) + np.log(x)
    
    def _outputLayerDelta(self, outputs, targets, activations):
        # training softmax on direct output delta is much more stable than putting through reverse function
        return outputs - targets
    
def encode(x):
    samples = x.shape[0]
    output_size = np.max(x) + 1
    
    encoded = np.zeros((int(samples), int(output_size)))
    
    for i, c in enumerate(x):
        if c == 0:
            encoded[i,0] = 1
            encoded[i,1] = 0
            encoded[i,2] = 0
        if c == 1:
            encoded[i,0] = 0
            encoded[i,1] = 1
            encoded[i,2] = 0
        if c == 2:
            encoded[i,0] = 0
            encoded[i,1] = 0
            encoded[i,2] = 1
            
    return encoded

        
"""
network = [{'inputs':2,  'nodes':20, 'activation': LeakyReLU()},
           {'inputs':20, 'nodes':20, 'activation': LeakyReLU()},
           {'inputs':20, 'nodes':20, 'activation': LeakyReLU()},
           {'inputs':20, 'nodes':3,  'activation':  Softmax()}]


test_nn = Network(structure=network, loss=ClassificationLoss())

inputs_train00 = np.ones((100,1)) + np.random.randn(100,1)
inputs_train01 = np.ones((100,1)) + np.random.randn(100,1)
targets0 = np.zeros((100,1))

inputs_train10 = 3*np.ones((100,1)) + np.random.randn(100,1)
inputs_train11 = np.ones((100,1)) + np.random.randn(100,1)
target1 = np.ones((100,1))

inputs_train20 = np.ones((100,1)) + np.random.randn(100,1)
inputs_train21 = 3*np.ones((100,1)) + np.random.randn(100,1)
target2 = np.ones((100,1)) * 2

X = np.zeros((300,3))
X[:,0] = np.concatenate([inputs_train00, inputs_train10, inputs_train20]).T
X[:,1] = np.concatenate([inputs_train01, inputs_train11, inputs_train21]).T
X[:,2] = np.concatenate([targets0, target1, target2]).T


np.random.shuffle(X)

test_nn.train()


for i in range(0,5000):
    print(f'epoch: {i}')
    test_nn.train()
    out = test_nn.forward(X[:,0:2], encode(X[:,2]))
    print('training loss:' + str(sum(test_nn.backprop.loss)))
    grad = test_nn.backward()
    test_nn.sgd(grad, lr=0.0001, lb=0)
    
    
test_nn.evaluate()

x11 = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100)
x22 = np.linspace(np.min(X[:,1]), np.max(X[:,1]), 100)

grid = np.meshgrid(x11,x22)
positions = np.vstack([grid[0].ravel(), grid[1].ravel()]).T
        
db_inputs = np.asarray(positions)
db_outs = test_nn.forward(db_inputs)

class1 = db_outs[:,0]
class2 = db_outs[:,1]
class3 = db_outs[:,2]

x0 = X[np.where(X[:,2]==0),:]
x1 = X[np.where(X[:,2]==1),:]
x2 = X[np.where(X[:,2]==2),:]
z1 = class1.reshape((100,100))
z2 = class2.reshape((100,100))
z3 = class3.reshape((100,100))

plt.scatter(x0[0][:,0], x0[0][:,1], color='y')
plt.scatter(x1[0][:,0], x1[0][:,1], color='b')
plt.scatter(x2[0][:,0], x2[0][:,1], color='r')

plt.contour(grid[0], grid[1], z1, levels=[0.6], colors=['yellow'])
plt.contour(grid[0], grid[1], z2, levels=[0.6], colors=['blue'])
plt.contour(grid[0], grid[1], z3, levels=[0.6], colors=['red'])
"""
        
