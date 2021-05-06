__author__ = "Yehya Albakri"

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from nnfs.datasets import sine
import time
from copy import copy, deepcopy


# Setting the seed to provide consistent results
# np.random.seed(1)


class Node():
    """ A Node object holds the value of the node after a forward pass, a delta generated after a backward pass, the previous nodes it's connected to, and their corresponding weights / bias. """

    def __init__(self, val=None, prev=None, prev_weight=None, next=None, bias=None, delta=None):
        self.val = val
        self.prev = prev
        self.prev_weight = prev_weight
        self.next = next
        self.bias = bias
        self.delta = delta


class Layer():
    """ When creating a layer object, the function sets all the variables from None to their corresponding empty value (either zero or empty list). The layer also stores the name of the activation function it uses (this will later be used in identifying the function in the forward and backward passes). """

    def __init__(self, length, activation_function):
        self.activation_function = activation_function
        self.length = length
        self.neurons = []
        for i in range(length):
            newNode = Node()
            if newNode.prev == None:
                newNode.prev = []
            if newNode.prev_weight == None:
                newNode.prev_weight = []
            if newNode.next == None:
                newNode.next = []
            if newNode.bias == None:
                newNode.bias = 0
            if newNode.delta == None:
                newNode.delta = 0
            self.neurons.append(newNode)

    def forward_relu(self):
        """ The relu function defines a linear function with a break in it. At x > 0, y = x. At x < 0, y = 0. It uses the values of the neurons in the current layer and passes it through the function. """
        temp_vals = []
        for neuron in self.neurons:
            for prevNeuron in neuron.prev:
                temp_vals.append(prevNeuron.val)

            # ReLU activation function in forward pass
            neuron.val = max(
                np.dot(neuron.prev_weight, temp_vals) + neuron.bias, 0)
            temp_vals = []

    def forward_none(self):
        """ This function essentially describes not having an activation function. But technically, it's a linear activation function. It outputs the sum of the sum of all the previous neurons' values multiplied by their corresponding weights. """
        temp_vals = []
        for neuron in self.neurons:
            for prevNeuron in neuron.prev:
                temp_vals.append(prevNeuron.val)

            # none activation function in forward pass
            neuron.val = np.dot(neuron.prev_weight, temp_vals) + neuron.bias
            temp_vals = []

    def forward_softmax(self):
        """ This function defines a softmax activation function. This is essentially exponentiates all the values in the layer, then takes the fraction of every exponentiated value over the sum of all the exponentiated values in the layer. The softmax activation function provides a normalized output, helpful in classification situations. """
        temp_vals_per_neuron = []
        temp_vals_all_neurons = []
        for neuron in self.neurons:
            for prevNeuron in neuron.prev:
                temp_vals_per_neuron.append(prevNeuron.val)

            temp_vals_all_neurons.append(
                np.dot(neuron.prev_weight, temp_vals_per_neuron) + neuron.bias)
            temp_vals_per_neuron = []

        # softmax activation funcion in forward pass
        # exp_values = np.exp(temp_vals_all_neurons -
        #                     np.max(temp_vals_all_neurons, axis=0, keepdims=True))
        exp_values = np.exp(temp_vals_all_neurons)
        # normalize exp_values
        exp_values_norm = exp_values / \
            np.sum(exp_values, axis=0, keepdims=True)
        # setting neuron values to softmax output
        for idx, neuron in enumerate(self.neurons):
            neuron.val = exp_values_norm[idx]


class Network():
    """ A Network object takes in the number of inputs, the number of hidden layers and their size, and the number of outputs and generates a network object. The object is essentially a series of fully connected layers of three types: the input, hidden layers, and the output. The user has the option to customize the activation functions of each layer (note: this is not a parameter, so it must be changed within the function). """

    def __init__(self, n_inputs, n_layers, n_layers_len, n_outputs):
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_layers_len = n_layers_len
        self.n_outputs = n_outputs

        self.nodes = []
        # creating input layer
        self.inputs = Layer(n_inputs, activation_function="relu")

        # creating the deep layers
        self.layers_deep = []
        for i in range(n_layers):
            self.layers_deep.append(
                Layer(n_layers_len, activation_function="relu"))

        # creating the output layer
        self.outputs = Layer(n_outputs, activation_function="relu")

        # connecting the input to the first deep layer (and initializing random weights)
        for i in self.inputs.neurons:
            for j in self.layers_deep[0].neurons:
                i.next.append(j)
                j.prev.append(i)
                j.prev_weight.append(0.01 * abs(np.random.randn()))

        # connecting the deep layers together (and initializing random weights)
        for i in range(n_layers-1):
            for neuron1 in self.layers_deep[i].neurons:
                for neuron2 in self.layers_deep[i+1].neurons:
                    neuron1.next.append(neuron2)
                    neuron2.prev.append(neuron1)
                    neuron2.prev_weight.append(0.01 * abs(np.random.randn()))

        # connecting the final deep layer to the output layer (and initializing random weights)
        for neuron in self.layers_deep[n_layers-1].neurons:
            for outputNeuron in self.outputs.neurons:
                neuron.next.append(outputNeuron)
                outputNeuron.prev.append(neuron)
                outputNeuron.prev_weight.append(0.01 * abs(np.random.randn()))

        # initializing random biases for the nuerons.
        for neuron in self.inputs.neurons:
            neuron.bias = 0.01 * abs(np.random.randn())
        for layer in self.layers_deep:
            for neuron in layer.neurons:
                neuron.bias = 0.01 * abs(np.random.randn())
        for neuron in self.outputs.neurons:
            neuron.bias = 0.01 * abs(np.random.randn())

    def print(self):
        """ The print function displays a graphical model of the neural network, representing the neurons as nodes and displaying the weights between the connections. This uses the NetworkX library. """
        G = nx.Graph()
        self.nodes = []

        # connecting the input to the first deep layer
        nodes = []
        for idx, i in enumerate(self.inputs.neurons):
            for j in self.layers_deep[0].neurons:
                G.add_edge(i, j, weight=round(j.prev_weight[idx], 5))
            nodes.append(i)
        self.nodes.append(nodes)

        # connecting the deep layers together
        for i in range(self.n_layers-1):
            for idx, neuron1 in enumerate(self.layers_deep[i].neurons):
                for neuron2 in self.layers_deep[i+1].neurons:
                    G.add_edge(neuron1, neuron2,
                               weight=round(neuron2.prev_weight[idx], 5))
        for layer in self.layers_deep:
            nodes = []
            for neuron in layer.neurons:
                nodes.append(neuron)
            self.nodes.append(nodes)

        # connecting the final deep layer to the output layer
        nodes = []
        for idx, neuron in enumerate(self.layers_deep[self.n_layers-1].neurons):
            for outputNeuron in self.outputs.neurons:
                G.add_edge(neuron, outputNeuron,
                           weight=round(outputNeuron.prev_weight[idx], 5))
        for outputNeuron in self.outputs.neurons:
            nodes.append(outputNeuron)
        self.nodes.append(nodes)

        # stores the graph inside the object
        self.graph = G

        # defines each layer in the network by a priority number (count). This is to maintain the order of the layers when displaying the graph.
        nx.set_node_attributes(self.graph, 0, "layer")
        count = 0
        for layer in self.nodes:
            count += 1
            for neuron in layer:
                self.graph.nodes[neuron]["layer"] = count

        pos = nx.multipartite_layout(self.graph, subset_key="layer")
        nx.draw(self.graph, pos, with_labels=False)
        labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph, pos, labels)

        # The following two lines display the graph for only a moment to create an animation of the visual changing over time.
        # plt.pause(0.00001)
        # plt.ion()

        # plt.figure()
        # plt.show()

    def forward(self, inputs):
        """ The forward function in the network runs the forward passes on all the layers. Note: make sure that the activation function here matches that defined in the init function. The init stores the name of the activation function to be used in the backward pass. This function stores the activation function to use in the forward pass. """
        for idx, value in enumerate(inputs):
            self.inputs.neurons[idx].val = value
        for layer in self.layers_deep:
            layer.forward_relu()
        self.outputs.forward_relu()

    def relu_derivative(self, val):
        """ Takes in a value and passes it through the relu activation derivative. """
        return (1 if val > 0 else 0)

    def softmax_derivative(self, val):
        """ Takes in a value and passes it through the softmax activation derivative. """
        return val * (1 - val)

    def backward(self, target_outputs, learning_rate, learning_rate_bias):
        """ The backward pass takes the gradient on the weights and biases and modifies them by a factor of the inputted learning rates. """

        # running the backward pass on the output layer
        errors = []
        for neuron, target_output in zip(self.outputs.neurons, target_outputs):
            errors.append((target_output - neuron.val))
            # print("error: ", errors[-1])
        for neuron, error in zip(self.outputs.neurons, errors):
            if self.outputs.activation_function == "relu":
                neuron.delta = error * self.relu_derivative(neuron.val)
            elif self.outputs.activation_function == "softmax":
                neuron.delta = error * self.softmax_derivative(neuron.val)
        # updating the weights of the output layer
        for neuron_output, neuron_final_hidden in zip(self.outputs.neurons, self.layers_deep[-1].neurons):
            neuron_output.prev_weight += learning_rate * \
                neuron_output.delta * neuron_final_hidden.val
            neuron_output.bias += learning_rate_bias * neuron_output.delta

        # storing the network's loss (at current output)
        self.loss = np.mean(errors)

        # running the backward pass on the final deep layer
        errors = []
        for idx_neuron_deep, neuron_deep in enumerate(self.layers_deep[-1].neurons):
            error = 0
            for neuron_output in self.outputs.neurons:
                error += neuron_output.prev_weight[idx_neuron_deep] * \
                    neuron_output.delta
            errors.append(error)
        for neuron, error in zip(self.layers_deep[-1].neurons, errors):
            if self.layers_deep[-1].activation_function == "relu":
                neuron.delta = error * self.relu_derivative(neuron.val)
            elif self.layers_deep[-1].activation_function == "softmax":
                neuron.delta = error * self.softmax_derivative(neuron.val)

        # running the backward pass on the rest of the deep layers
        for i in reversed(range(len(self.layers_deep)-1)):
            layer = self.layers_deep[i]
            next_layer = self.layers_deep[i+1]
            errors = []
            for idx, neuron in enumerate(layer.neurons):
                error = 0
                for right_neuron in next_layer.neurons:
                    error += right_neuron.prev_weight[idx] * right_neuron.delta
                errors.append(error)
            for neuron, error in zip(layer.neurons, errors):
                if layer.activation_function == "relu":
                    neuron.delta = error * self.relu_derivative(neuron.val)
                elif layer.activation_function == "softmax":
                    neuron.delta = error * self.softmax_derivative(neuron.val)

        # updating the weights of the deep layers' neurons
        for layer_idx, layer in enumerate(self.layers_deep):
            if layer_idx < len(self.layers_deep) - 1:
                for idx, neuron in enumerate(layer.neurons):
                    for next_idx, next_neuron in enumerate(self.layers_deep[layer_idx+1].neurons):
                        next_neuron.prev_weight[idx] += learning_rate * \
                            neuron.val * next_neuron.delta
                        next_neuron.bias += learning_rate_bias * neuron.delta


def main():
    # x, y = sine.create_data()
    x = []
    y = []
    for i in np.linspace(0, 1, 100):
        x.append([i])
        y.append([i*i])
        # y.append([0.25 * math.sin(i) + 0.5])

    b = Network(1, 2, 15, 1)
    # for neuron in b.outputs.neurons:
    #     for weight in neuron.prev_weight:
    #         print(id(weight))
    loss = []

    for i in range(1000):
        # print("Epoch: ", i)
        error = []
        lr = (1500 - i) / 1500
        print(lr)
        for i in range(len(x)):
            # if i % 50 == 0:
            #     b.print()
            b.forward(x[i])
            if i < 1000:
                b.backward(y[i], lr*0.1, lr*0.0001)
            elif i < 2000:
                b.backward(y[i], 0.001, 0.0001)
            elif i < 6000:
                b.backward(y[i], 0.005, 0.005)
            elif i < 8000:
                b.backward(y[i], 0.01, 0.001)
            elif i < 9000:
                b.backward(y[i], 0.05, 0.005)
            elif i < 10000:
                b.backward(y[i], 0.1, 0.01)
            elif i < 12000:
                b.backward(y[i], 2, 0.0001)
            elif i < 13000:
                b.backward(y[i], 3, 0.0001)
            elif i < 14000:
                b.backward(y[i], 5, 0.0001)
            elif i < 15000:
                b.backward(y[i], 10, 0.0001)
            elif i < 16000:
                b.backward(y[i], 50, 0.0001)
            elif i < 17000:
                b.backward(y[i], 100, 0.0001)
            elif i < 18000:
                b.backward(y[i], 250, 0.0001)
            elif i < 19000:
                b.backward(y[i], 500, 0.0001)
            elif i < 20000:
                b.backward(y[i], 1000, 0.0001)
            error.append(b.loss)
        loss.append(np.mean(error))

    X = []
    Y = []
    for i in np.linspace(0, 1, 100):
        X.append(i)
        b.forward([i])
        Y.append(deepcopy(b.outputs.neurons[0].val))
    plt.plot(X, Y, label="Neural Network Output")
    plt.plot(x, y, label="Training Data")
    plt.title("Network Output vs Training Data")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.figure()

    b.print()
    # plt.title("Neural Network Node Plot")
    # plt.figure()

    ax = plt.figure()
    ax = ax.add_subplot(1, 1, 1)
    line, = ax.plot(loss)
    # ax.set_yscale('log')
    plt.title("Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.show()
    # print(x)
    # print(y)


if __name__ == '__main__':
    pass
    main()

""" 
TODO:
$ = DONE
# = NEVERMIND

$ clear neuron delta after each backward pass
# Implement batch learning
# implement dropout layers
# implement normalization / preprocessing
$ implement dynamic learning rate
$ Comment code
$ fix print function
$ store loss and accuracy
- fix print function getting slower over time
"""
