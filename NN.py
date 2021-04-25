__author__ = "Yehya Albakri"

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from nnfs.datasets import sine
import time
from copy import copy, deepcopy

np.random.seed(1)


# create neuron (Node)
class Node():
    def __init__(self, val=None, prev=None, prev_weight=None, next=None, bias=None, delta=None):
        self.val = val
        self.prev = prev
        self.prev_weight = prev_weight
        self.next = next
        self.bias = bias
        self.delta = delta


class Layer():
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
        # calculate new neuron value
        temp_vals = []
        for neuron in self.neurons:
            for prevNeuron in neuron.prev:
                temp_vals.append(prevNeuron.val)

            # ReLU activation function in forward pass
            neuron.val = max(
                np.dot(neuron.prev_weight, temp_vals) + neuron.bias, 0)
            temp_vals = []

    def forward_none(self):
        # calculate new neuron value
        temp_vals = []
        for neuron in self.neurons:
            for prevNeuron in neuron.prev:
                temp_vals.append(prevNeuron.val)

            # none activation function in forward pass
            neuron.val = np.dot(neuron.prev_weight, temp_vals) + neuron.bias
            temp_vals = []

    def forward_softmax(self):
        temp_vals_per_neuron = []
        temp_vals_all_neurons = []
        for neuron in self.neurons:
            for prevNeuron in neuron.prev:
                temp_vals_per_neuron.append(prevNeuron.val)

            temp_vals_all_neurons.append(
                np.dot(neuron.prev_weight, temp_vals_per_neuron) + neuron.bias)
            temp_vals_per_neuron = []

        # softmax activation funcion in forward pass
        exp_values = np.exp(temp_vals_all_neurons -
                            np.max(temp_vals_all_neurons, axis=0, keepdims=True))
        # normalize exp_values
        exp_values_norm = exp_values / \
            np.sum(exp_values, axis=0, keepdims=True)
        # setting neuron values to softmax output

        for idx, neuron in enumerate(self.neurons):
            neuron.val = exp_values_norm[idx]


class Network():
    def __init__(self, n_inputs, n_layers, n_layers_len, n_outputs):
        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_layers_len = n_layers_len
        self.n_outputs = n_outputs

        # G = nx.Graph()
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

        # connecting the input to the first deep layer
        # nodes = []
        for i in self.inputs.neurons:
            for j in self.layers_deep[0].neurons:
                i.next.append(j)
                j.prev.append(i)
                j.prev_weight.append(0.01 * abs(np.random.randn()))
        #         G.add_edge(i, j, weight=round(j.prev_weight[-1], 5))
        #     nodes.append(i)
        # self.nodes.append(nodes)

        # connecting the deep layers together
        for i in range(n_layers-1):
            for neuron1 in self.layers_deep[i].neurons:
                for neuron2 in self.layers_deep[i+1].neurons:
                    neuron1.next.append(neuron2)
                    neuron2.prev.append(neuron1)
                    neuron2.prev_weight.append(0.01 * abs(np.random.randn()))
                    # G.add_edge(neuron1, neuron2,
                    #            weight=round(neuron2.prev_weight[-1], 5))
        # for layer in self.layers_deep:
        #     nodes = []
        #     for neuron in layer.neurons:
        #         nodes.append(neuron)
        #     self.nodes.append(nodes)

        # connecting the final deep layer to the output layer
        # nodes = []
        for neuron in self.layers_deep[n_layers-1].neurons:
            for outputNeuron in self.outputs.neurons:
                neuron.next.append(outputNeuron)
                outputNeuron.prev.append(neuron)
                outputNeuron.prev_weight.append(0.01 * abs(np.random.randn()))
                # G.add_edge(neuron, outputNeuron,
                #            weight=round(outputNeuron.prev_weight[-1], 5))

        # for outputNeuron in self.outputs.neurons:
        #     nodes.append(outputNeuron)
        # self.nodes.append(nodes)
        # self.graph = G

    def print(self):
        G = nx.Graph()
        self.nodes = []

        # connecting the input to the first deep layer
        nodes = []
        for i in self.inputs.neurons:
            for j in self.layers_deep[0].neurons:
                G.add_edge(i, j, weight=round(j.prev_weight[-1], 5))
            nodes.append(i)
        self.nodes.append(nodes)

        # connecting the deep layers together
        for i in range(self.n_layers-1):
            for neuron1 in self.layers_deep[i].neurons:
                for neuron2 in self.layers_deep[i+1].neurons:
                    G.add_edge(neuron1, neuron2,
                               weight=round(neuron2.prev_weight[-1], 5))
        for layer in self.layers_deep:
            nodes = []
            for neuron in layer.neurons:
                nodes.append(neuron)
            self.nodes.append(nodes)

        # connecting the final deep layer to the output layer
        nodes = []
        for neuron in self.layers_deep[self.n_layers-1].neurons:
            for outputNeuron in self.outputs.neurons:
                G.add_edge(neuron, outputNeuron,
                           weight=round(outputNeuron.prev_weight[-1], 5))
        for outputNeuron in self.outputs.neurons:
            nodes.append(outputNeuron)
        self.nodes.append(nodes)

        self.graph = G

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
        plt.pause(0.00001)
        plt.ion()
        plt.show()

    def forward(self, inputs):
        for idx, value in enumerate(inputs):
            self.inputs.neurons[idx].val = value
        for layer in self.layers_deep:
            layer.forward_relu()
        self.outputs.forward_relu()

    def relu_derivative(self, val):
        return (1 if val > 0 else 0)
        # return 1

    def softmax_derivative(self, val):
        return val * (1 - val)

    # def loss(self, target_output):
    #     self.loss = 0
    #     for neuron, target_val in zip(self.outputs.neurons, target_output):
    #         # Note: neuron.val is the softmax output of the output layer
    #         self.loss += math.log(neuron.val) * target_val
    #     self.loss *= -1

    def backward(self, target_outputs, learning_rate):
        # running the backward pass on the output layer
        errors = []
        for neuron, target_output in zip(self.outputs.neurons, target_outputs):
            errors.append(target_output - neuron.val)
        # print(errors)
        for neuron, error in zip(self.outputs.neurons, errors):
            if self.outputs.activation_function == "relu":
                neuron.delta = error * self.relu_derivative(neuron.val)
            elif self.outputs.activation_function == "softmax":
                neuron.delta = error * self.softmax_derivative(neuron.val)
        # updating the weights of the output layer
        for neuron_output, neuron_final_hidden in zip(self.outputs.neurons, self.layers_deep[-1].neurons):
            neuron_output.prev_weight += learning_rate * \
                neuron_output.delta * neuron_final_hidden.val
            neuron_output.bias += 0.001 * learning_rate * neuron_output.delta

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
                # for neuron, next_neuron in zip(layer.neurons, self.layers_deep[layer_idx+1].neurons):
                for idx, neuron in enumerate(layer.neurons):
                    # neuron.prev_weight += learning_rate * neuron.delta * next_neuron.val
                    for next_idx, next_neuron in enumerate(self.layers_deep[layer_idx+1].neurons):
                        next_neuron.prev_weight[idx] += learning_rate * \
                            neuron.val * next_neuron.delta
                        next_neuron.bias += 0.001 * learning_rate * neuron.delta
                    # neuron.bias += learning_rate * neuron.delta


def main():
    pass
    x, y = sine.create_data()
    x = []
    y = []
    for i in np.linspace(0, 1, 100):
        x.append([i])
        y.append([i*i])

    b = Network(1, 2, 100, 1)
    # b.print()
    print("stage1", b.nodes)
    # nx.set_node_attributes(b.graph, 0, "layer")
    # count = 0
    # for layer in b.nodes:
    #     count += 1
    #     for neuron in layer:
    #         b.graph.nodes[neuron]["layer"] = count
    # pos = nx.multipartite_layout(b.graph, subset_key="layer")
    # nx.draw(b.graph, pos, with_labels=False)
    # plt.show()
    # b.forward([0.5])
    # b.backward([-0.5], 0.01)
    # for neuron in b.outputs.neurons:
    #     print("prev weights", neuron.prev_weight)
    #     print("val", neuron.val)
    #     print("delta", neuron.delta)
    # b.backward([0], 1)
    # b.forward([0.2])
    # for neuron in b.outputs.neurons:
    #     print(neuron.prev_weight)
    #     print(neuron.val)

    for i in range(3000):
        for i in range(len(x)):
            # for i in range(10):
            # if i % 50 == 0:
            #     b.print()
            b.forward(x[i])
            if i < 2000:
                b.backward(y[i], 0.1)
            else:
                b.backward(y[i], 0.03)

    # for i in range(50):
    #     # b.print()
    #     # print(1)
    #     b.forward([1])
    #     b.backward([1], 1)

    # for neuron in b.layers_deep[0].neurons:
    #     print(neuron.delta)

    # b.forward([0.5])
    # for neuron in b.outputs.neurons:
    #     print(neuron.prev_weight)
    #     print(neuron.val)
    #     print(neuron.delta)

    # b.forward([1])
    # for neuron in b.outputs.neurons:
    #     print(neuron.prev_weight)
    #     print(neuron.val)
    #     print(neuron.delta)

    X = []
    Y = []
    for i in np.linspace(0, 1, 100):
        X.append(i)
        b.forward([i])
        Y.append(deepcopy(b.outputs.neurons[0].val))
    plt.plot(X, Y)
    plt.plot(x, y)
    plt.show()
    # print(x)
    # print(y)


if __name__ == '__main__':
    pass
    # Network(3, 3, 10, 3)
    # print(np.random.randn())
    main()

""" 
TODO:

- clear neuron delta after each backward pass
"""
