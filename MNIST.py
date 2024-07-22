from container import ImageFile, LabelFile
import basicNN
import numpy as np
import matplotlib

training_images = np.array(ImageFile('data/train-images').read())
training_labels = np.array(LabelFile('data/train-labels').read())
validation_images = np.array(ImageFile('data/t10k-images').read())
validation_labels = np.array(LabelFile('data/t10k-labels').read())

network = [{'inputs':784,'nodes':784, 'activation': basicNN.LeakyReLU()},
           {'inputs':784,'nodes':300, 'activation': basicNN.LeakyReLU()},
           {'inputs':300,'nodes':100, 'activation': basicNN.LeakyReLU()},
           {'inputs':100,'nodes': 50, 'activation': basicNN.LeakyReLU()},
           {'inputs':50, 'nodes':10,  'activation':  basicNN.Softmax()}]

nn = basicNN.Network(structure=network, loss=basicNN.ClassificationLoss())

#normalize pixel values from 0 to 1
if max(validation_images[0]) > 1: validation_images = validation_images / 255
if max(training_images[0]) > 1: training_images = training_images / 255

#split into batches
batch_size = 128

training_inputs_batches = np.array_split(training_images, len(training_images) // batch_size)
training_labels_batches = np.array_split(training_labels, len(training_labels) // batch_size)

validation_inputs_batches = np.array_split(validation_images, len(validation_images) // batch_size)
validation_labels_batches = np.array_split(validation_labels, len(validation_labels) // batch_size)

epochs = 5

aloss = []
time = []

for epoch in range(epochs):
    print(f'epoch: {epoch}')
    
    #training
    nn.train()
    for (inputs, targets) in zip(training_inputs_batches, training_labels_batches):
        out = nn.forward(inputs, targets)
        loss = sum(sum(nn.backprop.loss)) / batch_size
        print('training loss:' + str(loss))
        grad = nn.backward()
        nn.sgd(grad, lr=0.0001, lb=0.1)
        
        aloss.append(loss)
        
        if time == []:
            time.append(1)
        else:
            time.append(time[-1] + 1)
    
        matplotlib.pyplot.plot(time, aloss, color='b')
        matplotlib.pyplot.show()
        
    matplotlib.pyplot.close()
    aloss = []
    time = []    
    
    #validation
    for (inputs, targets) in zip(validation_inputs_batches, validation_labels_batches):
        out = nn.forward(inputs, targets)
        loss = sum(sum(nn.backprop.loss)) / batch_size
        print('validation loss:' + str(loss))
        
        aloss.append(loss)
        
        if time == []:
            time.append(1)
        else:
            time.append(time[-1] + 1)
    
        matplotlib.pyplot.plot(time, aloss, color='b')
        matplotlib.pyplot.show()
        
    matplotlib.pyplot.close()
    aloss = []
    time = [] 
    
nn.evaluate()
correct = 0 
incorrect = 0

# percentage correct of validations set  
for (inputs, targets) in zip(validation_inputs_batches, validation_labels_batches):
    out = nn.forward(inputs)
    
    for i in range(len(targets)):
        idx = np.where(targets[i] == True)[0]
        if out[i][idx][0] > 0.5:
            correct += 1
        else:
            incorrect += 1
            
print(correct/10000)
    

        