import numpy as np


class HiddenLayer:
    def __init__(self, input_size, number_of_hidden_neurons, rect=True, reg=0.1, step_size=0.01):
        self.step_size = step_size
        self.rect = rect
        self.reg = reg
        self.zero = np.zeros([number_of_hidden_neurons, 1])
        self.w = np.random.rand(number_of_hidden_neurons, input_size) - 0.5
        self.b = np.random.rand(number_of_hidden_neurons, 1) - 0.5
        self.input = None
        self.out = None

    def update(self, dw, db):
        # Regularization: if self.reg = 0, no regularization
        dw = dw - self.reg * self.w
        self.w = self.w + self.step_size * dw
        self.b = self.b + self.step_size * db

    def forward_pass(self, x):
        self.input = x
        self.out = np.matmul(self.w, self.input) + self.b
        if self.rect:
            self.out = np.maximum(0, self.out)
        return self.out

    def backward_pass(self, grad):
        if self.rect:
            grad = np.where(self.out == 0, self.zero, grad)
        dw = np.outer(grad, self.input)
        db = grad
        dinput = np.matmul(np.transpose(self.w), grad)
        self.update(dw, db)
        return dinput


class NN:
    def __init__(self, input_size, hidden_layer_sizes=None, reg=0.1, step_size=0.01):
        self.input_size = input_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.reg = reg
        self.step_size = step_size
        self.layers = []
        in_size = self.input_size
        if self.hidden_layer_sizes is not None:
            for i in range(len(self.hidden_layer_sizes)):
                li = HiddenLayer(in_size, self.hidden_layer_sizes[i], rect=True, reg=self.reg, step_size=self.step_size)
                self.layers.append(li)
                in_size = self.hidden_layer_sizes[i]
        l_out = HiddenLayer(in_size, 1, rect=False, reg=self.reg, step_size=self.step_size)
        self.layers.append(l_out)

    def forward_pass(self, x):
        x = np.reshape(x, [self.input_size, 1])
        for l in self.layers:
            x = l.forward_pass(x)
        return x

    def update(self, grad):
        grad = np.reshape([grad], [1, 1])
        for l in self.layers[::-1]:
            grad = l.backward_pass(grad)

    def accuracy(self, X, Y):
        n_corr = 0
        for i in range(len(Y)):
            out = self.forward_pass(X[i])
            prediction = 1
            if out < 0:
                prediction = -1
            if prediction == Y[i]:
                n_corr += 1
        return n_corr/len(Y)

    def train(self, X, Y, epoch):
        for ep in range(epoch):
            for i in range(len(Y)):
                out = self.forward_pass(X[i])
                grad = 0.0
                if Y[i] == 1 and out < 1:
                    grad = 1.0
                elif Y[i] == -1 and out > -1:
                    grad = -1.0
                self.update(grad)
            if ep % 5 == 0:
                print('training accuracy at epoch ' + str(ep) + ': ' + str(self.accuracy(X, Y)))

    def predict(self, X):
        Y = []
        for i in range(len(X)):
            out = self.forward_pass(X[i])
            prediction = 1
            if out < 0:
                prediction = -1
            Y.append(prediction)
        return Y

#########################################################################################
# Choose number of hidden layers and size of each hidden layer
# For example, if you want two hidden layers of sizes 10 and 5:
# hidden_layers = [10, 5]
# If you don't want any hidden layers:
# hidden_layers = None
# One hidden layer with 5 neurons:
hidden_layers = [5]
# As you increase number/size of hidden layers, you might want to increase iterations
iterations = 1000
# Also try tweaking step_size
step_size = 0.01
#########################################################################################

X_train = []
Y_train = []
# In train.txt file, each line contains one example
# Each example consists of numbers separated by ONLY comma (,)
# First number of example is class indicator (1 or -1)
# Rest of the numbers are features
# Each example must have class indicator and same number of features
with open('train.txt') as f:
    for line in f:
        data = [float(x.strip()) for x in line.split(',')]
        Y_train.append(data[0])
        X_train.append(data[1:])
print(X_train)
print(Y_train)

nn = NN(len(X_train[0]), hidden_layers, step_size=step_size)
print(nn.accuracy(X_train, Y_train))
nn.train(X_train, Y_train, iterations)

X_test = []
# In test.txt file, each line contains one example
# Each example consists of numbers separated by ONLY comma (,)
# All of the numbers are features
# Each example must have same number of features
with open('test.txt') as f:
    for line in f:
        X_test.append([float(x.strip()) for x in line.split(',')])
print(X_test)
print(nn.predict(X_test))