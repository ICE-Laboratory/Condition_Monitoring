""" 
- The creation of dataset is taken from: https://cs231n.github.io/neural-networks-case-study/

"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# This function generates a spiral dataset
def create_data():
    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype="uint8")  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    return X, y


# input of neurons
X, y = create_data()

# Creates a hidden layer in neural network
#
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Creating a random weight with specified matrix size
        # n_inputs = number of inputs
        # n_neurons = number of neurons
        # randn: gaussian distribution rounded around 0
        # We dont need to transpose the weight when doing dot product
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons
        )  # Multiplying by 0*10 to achieve values less than 1
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights + self.biases)


# Activation of each neurons in the hidden layer
class Activation_ReLu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# X, y = 2 inputs
# We created 5 neurons in the first hidden layer
layer1 = Layer_Dense(2, 5)
#
activation1 = Activation_ReLu()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)
