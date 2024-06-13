from container import ImageFile, LabelFile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from neuralnet import Network

training_images = ImageFile('data/train-images')
training_labels = LabelFile('data/train-labels')
validation_images = ImageFile('data/t10k-images')
validation_labels = LabelFile('data/t10k-labels')

training_images.read()
training_labels.read()
validation_images.read()
validation_labels.read()


run_torch = False
if run_torch:
    X_train, Y_train = torch.Tensor(training_images.get_array()), torch.Tensor(training_labels.get_array())
    X_validate, Y_validate = torch.Tensor(validation_images.get_array()), torch.Tensor(validation_labels.get_array())

    train_dataset = TensorDataset(X_train, Y_train)
    validate_dataset = TensorDataset(X_validate, Y_validate)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    validate_loader = DataLoader(validate_dataset, batch_size=64, shuffle=True)

    class Model(nn.Module):
        def __init__(self, x_size, y_size, hl1_size, hl2_size):
            super(Model, self).__init__()
            self.fc1 = nn.Linear(x_size, hl1_size)
            self.fc2 = nn.Linear(hl1_size, hl2_size)
            self.fc3 = nn.Linear(hl2_size, y_size)
            self.reLU = nn.ReLU()

        def forward(self, x):
            x = self.fc1(x)
            x = self.reLU(x)
            x = self.fc2(x)
            x = self.reLU(x)
            x = self.fc3(x)
            return x
        
    model = Model(x_size=len(X_train[0]), y_size=len(Y_train[0]), hl1_size=500, hl2_size=500)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    epochs = 10
    for epoch in range(epochs):

        model.train()
        for inputs, target in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            correct = 0
            incorrect = 0
            omitted = 0
            
            for inputs, target in validate_loader:
                outputs = model(inputs)
                outputs_norm = outputs.softmax(dim=1)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()

                for index1, output_arr in enumerate(outputs_norm):
                    for index2, output in enumerate(output_arr):
                        if target[index1][index2]:
                            if output > 0.5:
                                correct += 1
                            else:
                                incorrect += 1

            average_val_loss = val_loss / len(validate_loader)

            print(f'Validation Loss: {average_val_loss}, Correct: {correct}, Incorrect: {incorrect}')


run_neuralnet = True
if run_neuralnet:



print("stop")