"""
This code represents a simple neural network script from scratch without classes. 

-Pretedetermined inputs, weights and biases with 2 hidden layers were used
"""

import numpy as np

# Inputs represents the neurons
# Each inputs can be considered as sensors of our system (features of our samples)
inputs = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]
# Every neurons have their own weights and Biases
weights = [[0.2, 0.8, -0.5, 1.0], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]]
# Second layer
weights2 = [[0.1, -0.14, -0.5], [-0.5, -0.12, 0.33], [-0.44, -0.73, -0.13]]
# bias: is an offset of neuron's output value
biases = [2, 3, 0.5]
# layer2 biases
biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)


"""
- Applying dot product manually

layer_output = []

# zip combines two lists elementwise
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
"""
