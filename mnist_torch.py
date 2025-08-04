import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

''' Basic MNIST classification using dense layers with torch '''

# hyperparameters overview
input_size = 784 #(= 28x28)
output_size = 10 # (0 to 9)
hidden_size = 100 # or whatever
epochs = 20 # or whatever
learning_rate = 5e-5 # usually 10-3 or 10-5
batch_size = 50 # size of mini batches (during training)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        return F.log_softmax(x, dim=0)
    
def load_data(train_path, test_path):
    # training data (60000)
    df_train_data = pd.read_csv(train_path, sep=',')
    df_train_values = df_train_data.drop('label', axis=1)
    df_train_labels = df_train_data['label']
    
    # I'll use numpy here
    train_values = df_train_values.to_numpy()
    train_labels = df_train_labels.to_numpy()
    
    # test data (10000), same deal
    df_test_data = pd.read_csv(test_path, sep=',')
    df_test_values = df_test_data.drop('label', axis=1)
    df_test_labels = df_test_data['label']
    
    test_values = df_test_values.to_numpy()
    test_labels = df_test_labels.to_numpy()
    
    return train_values, train_labels, test_values, test_labels

if __name__ == '__main__':
    cwd = os.getcwd()
    train_path = os.path.join(cwd, 'data', 'mnist_train.csv')
    test_path = os.path.join(cwd, 'data', 'mnist_test.csv')
    train_values, train_labels, test_values, test_labels = load_data(train_path, test_path)
    
    # torch stores data as tensors
    x = torch.FloatTensor(train_values.tolist())
    y = torch.LongTensor(train_labels.tolist())
    
    net = Network()
    
    # stochastic gradient descent with momentum
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    loss_func = nn.CrossEntropyLoss() # why not
    
    loss_log = [] # keep track
    for epoch in range(epochs):
        for i in range(0, x.shape[0], batch_size):
            x_batch = x[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            
            x_var = Variable(x_batch)
            y_var = Variable(y_batch)
            
            optimizer.zero_grad() # resets the optimizer
            out = net(x_var) # forward pass
            
            loss = loss_func(out, y_var)
            loss.backward() # backward prop
            optimizer.step() # update
            
            if i % 100 == 0:
                loss_log.append(loss.item())
        
        print(f'Epoch {epoch} - Loss: {loss.item()}')
    
    plt.plot(loss_log)
    plt.show()
    
    # train accuracy
    X = torch.FloatTensor(train_values.tolist())
    X = Variable(X)
    predictions_train = torch.max(net(X).data, 1)[1].numpy()
    
    # test
    X = torch.FloatTensor(test_values.tolist())
    X = Variable(X)
    predictions_test = torch.max(net(X).data, 1)[1].numpy()
    
    # final scores
    score_train = np.round(np.mean(train_labels == predictions_train), 5)
    score_test = np.round(np.mean(test_labels == predictions_test), 5)
    print(f'Training accuracy: {score_train} | test accuracy: {score_test}')