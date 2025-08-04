import numpy as np
import pandas as pd
import os

class Dense:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        
        self.w = np.random.randn(output_size, input_size) * np.sqrt(2./input_size)
        self.b = np.zeros((output_size, 1))
        
    def __repr__(self):
        return f'Dense ({self.input_size}, {self.output_size})'
    
    def forward(self, x: np.array):
        self.input = x # keep for backprop
        return self.w.dot(x) + self.b
    
    def backward(self, output_gradient: np.array):
        # gradients
        dw = output_gradient.dot(self.input.T)
        db = output_gradient
        da = self.w.T.dot(output_gradient)
        
        return da, dw, db

class ReLu:
    def __init__(self):
        self.f = lambda x: np.maximum(0,x)
        self.df = lambda x: np.array(x>0, dtype=float)
        
    def __repr__(self):
        return "ReLu"
    
    def forward(self, x: np.array):
        self.input = x
        return self.f(x)
    
    def backward(self, output_gradient: np.array):
         # dC/dx = dC/da * da/dx = grad * a'(x)
         # this result is actually valid for any activation layer (tanh, sigmoid, etc...)
        return output_gradient * self.df(self.input)

def softmax(x: np.array):
    # the softmax version exp / sum(exp) is numerically unstable
    # if you roll with this, the values explodes and you get nothing
    # the solution is the safe softmax, which guarantees exp below 1
    exps = np.exp(x - np.max(x))
    return exps/np.sum(exps, axis=0)

# the followings are just basic loss functions and their derivatives
def cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon) # to avoid log(0)
    return -np.sum(y_true * np.log(y_pred))

def d_cross_entropy(y_true, y_pred):
    return y_pred - y_true

def mse(y_true, y_pred):
    return np.mean((y_true-y_pred)**2)

def d_mse(y_true, y_pred):
    return 2*(y_pred-y_true)/len(y_true)

def to_onehot(labels, size=10):
    onehot = np.zeros((len(labels), 10, 1))
    for i, y in enumerate(labels):
        onehot[i, y] = 1.0
    labels = onehot
    return onehot

def load_data(train_path, test_path):
    # training data (60000)
    df_train_data = pd.read_csv(train_path, sep=',')
    df_train_values = df_train_data.drop('label', axis=1)
    df_train_labels = df_train_data['label']
    
    # I'll use numpy here
    train_values = df_train_values.to_numpy()
    train_labels = df_train_labels.to_numpy()
    
    # training data (60000)
    train_values = train_values / 255.0
    train_values = train_values.reshape(train_values.shape[0], 784, 1)
    train_labels = to_onehot(train_labels)
    
    # test data (10000), same deal
    df_test_data = pd.read_csv(test_path, sep=',')
    df_test_values = df_test_data.drop('label', axis=1)
    df_test_labels = df_test_data['label']
    
    test_values = df_test_values.to_numpy()
    test_labels = df_test_labels.to_numpy()
    
    test_values = test_values / 255.0
    test_values = test_values.reshape(test_values.shape[0], 784, 1)
    test_labels = to_onehot(test_labels)
    
    return train_values, train_labels, test_values, test_labels

def compute_accuracy(net, values, labels):
    accuracy = 0
    for x, y in zip(values, labels):
        x = x.reshape(784,1)
        output = x
        for layer in net:
            output = layer.forward(output)
    
        if np.argmax(softmax(output)) == np.argmax(y):
            accuracy += 1
    return accuracy / len(values)

def train(net: list, values: np.array, labels: np.array, epochs=20, learning_rate=1e-3, batch_size=50):
    for epoch in range(epochs):
        error = 0
        
        for i in range(0, len(values), batch_size):
            x_batch = values[i:i+batch_size]
            y_batch = labels[i:i+batch_size]
            
            # keep track of all gradients for later SGD
            grads = [ {'dw': 0, 'db': 0} for layer in net if isinstance(layer, Dense) ]
            
            for x, y in zip(x_batch, y_batch):
                # forward pass
                output = x
                for layer in net:
                    output = layer.forward(output)
                
                y_pred = softmax(output)
                
                # mse gives (predictably) worse results
                error += cross_entropy(y, y_pred)
                grad = d_cross_entropy(y, y_pred)
                # error += mse(y, y_pred)
                # grad = dmse(y, y_pred)
                
                # backward pass
                layer_grads = []
                for layer in reversed(net):
                    # for every dense layers, gather the gradients
                    if (isinstance(layer, Dense)):
                        grad, dw, db = layer.backward(grad)
                        layer_grads.append((dw, db))
                    else:
                        grad = layer.backward(grad)
                
                # increment the gradient
                # is is basically a running average calculation
                # I moved the division here for consistency, although is it inefficient (but this is Python, so who cares)
                for idx, (dw, db) in enumerate(reversed(layer_grads)):
                    grads[idx]['dw'] += dw / batch_size
                    grads[idx]['db'] += db / batch_size
            
            # for every dense layer, grab the gradients and update the parameters
            idx = 0
            for layer in net:
                if isinstance(layer, Dense):
                    layer.w -= learning_rate * grads[idx]['dw']
                    layer.b -= learning_rate * grads[idx]['db']
                    idx += 1
        
        # some results
        error /= len(train_values) * batch_size
        accuracy = compute_accuracy(net, train_values, train_labels)
        print(f"Epoch {epoch} | error: {error:.4f} | accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    # hyperparameters
    input_size = 784 # (= 28x28)
    output_size = 10 # (0 to 9)
    hidden_size = 100 # or whatever
    epochs = 20 # or whatever
    learning_rate = 1e-3 # usually 10-3 or 10-5
    batch_size = 20 # size of mini batches (during training)
    
    # data
    cwd = os.getcwd()
    train_path = os.path.join(cwd, 'data', 'mnist_train.csv')
    test_path = os.path.join(cwd, 'data', 'mnist_test.csv')
    train_values, train_labels, test_values, test_labels = load_data(train_path, test_path)
    
    # network and training
    net = [
        Dense(input_size, hidden_size),
        ReLu(),
        Dense(hidden_size, output_size)
    ]
    train(net, train_values, train_labels, epochs, learning_rate, batch_size)
    
    # let's test it
    test_accuracy = compute_accuracy(net, test_values, test_labels)
    print(f'Test accuracy: {np.mean(test_accuracy)}')
    
    