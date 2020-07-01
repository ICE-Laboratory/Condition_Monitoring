import numpy as np
import matplotlib.pyplot as plt


class Gradient_Descent:
    def __init__(self, x, target, weights, learning_rate):
        self.x = x
        self.target = target
        self.learning_rate = learning_rate
        self.w = weights

    def backward_pass(self, predicted, target, x):
        # Backward Pass
        self.predicted = predicted
        self.g1 = 2 * (self.predicted - self.target)  # (d)error / (d)predicted
        self.g2 = (  # (d)predicted / (d) sop
            1.0
            / (1 + np.exp(-1 * self.predicted))
            * (1 - (1.0 / (1 + np.exp(-1 * self.predicted))))
        )
        self.g3 = self.x  # (d) sop / (d) W
        self.grad = self.g3 * self.g2 * self.g1  # (d) error / (d) W
        self.err = np.power(
            self.predicted - self.target, 2
        )  # Error calculation between predcited vs actual
        self.w = self.w - self.learning_rate * self.grad  # Updating the weights
        return self.w  # Returning th updated weights


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        # Creating a random weight with specified matrix size
        # n_inputs = number of inputs
        # n_neurons = number of neurons
        # randn: gaussian distribution rounded around 0
        # We dont need to transpose the weight when doing dot product
        # self.weights = 0.2
        self.weights = 0.10 * np.random.randn(
            n_inputs, n_neurons
        )  # Multiplying by 0*10 to achieve values less than 1
        # self.biases = 0
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        # Forward Pass
        self.sop = np.dot(inputs, self.weights) + self.biases  # Calculation of SOP

    # Activation of each neurons in the hidden layer
    def activate(self):
        self.output = 1.0 / (
            1 + np.exp(-1 * self.sop)
        )  # Activation of neuron with sigmoid function


# Initialize the dimensions and iteration
n_iter = 80000  # Number of iterations
n_neurons = 2  # Number of neurons on the hidden layer
d_inputs = 1  # the dimension of the inputs

# Initialize the values
x = 0.1  # The input value
target = 0.3  # The target that we are approximating
learning_rate = 0.01  # Learning rate of GD


# Input, Output, Learning rate are being fed
layer1 = Layer_Dense(d_inputs, n_neurons)  # A hidden layer has been created
layer1.forward(x)  # SOP (sum of all products) has been calculated of neurons
layer1.activate()  # The calculation of predicted output by activain the neurons
GD = Gradient_Descent(
    x, target, layer1.weights, learning_rate
)  # GD instance is initialized
layer1.weights = GD.backward_pass(
    layer1.output, target, x
)  # the new weight is calculated by backward pass


y_axis = np.zeros((n_iter, n_neurons))  # Creating the y axis for graph
# The iterative optimization for weight tuning
for k in range(n_iter):
    # Forward pass of neurons
    layer1.forward(x)
    layer1.activate()
    # Updating the weights by backward pass
    layer1.weights = GD.backward_pass(layer1.output, target, x)
    y_axis[k, :] = GD.err  # passing error values to plot the graph


# Plotting the results (error vs # of iterations)
x_axis = np.arange(k + 1)  # x_axis is created (n_itearions)
# Creating subplots
fig, axs = plt.subplots(n_neurons, sharex=True, sharey=True)
fig.suptitle(
    "Error vs Number of Iterations with Learning Rate = {}".format(learning_rate)
)

# Plotting each neurons
for k in range(1):
    for k in range(n_neurons):
        axs[k].set_xlabel("Number of Itearion")
        axs[k].set_ylabel("Error")
        axs[k].plot(x_axis, y_axis[:, k])
        axs[k].set_title("Neuron: {}".format(k + 1))
    plt.show()
