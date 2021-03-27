__author__ = "Yehya Albakri"

import numpy as np

# create neuron (Node)


class Node():
    def __init__(self, val=None, prev=None, prev_weight=None, next=None, bias=None):
        self.val = val
        self.prev = prev
        self.prev_weight = prev_weight
        self.next = next
        self.bias = bias


class Layer():
    def __init__(self, length):
        self.length = length
        self.neurons = set()
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
            self.neurons.add(newNode)

    def forward(self, inputs):
        # saving the input values
        self.inputs = inputs

        # calculate new neuron value
        temp_vals = []
        for neuron in self.neurons:
            for prevNeuron in neuron.prev:
                temp_vals.append(prevNeuron.val)
            neuron.val = max(
                np.dot(neuron.prev_weight, temp_vals) + neuron.bias, 0)
            temp_vals = []


class Network():
    def __init__(self, n_inputs, n_layers, n_layers_len, n_outputs):
        # creating input layer
        self.inputs = Layer(n_inputs)
        # creating the deep layers
        self.layers_deep = []
        for i in range(n_layers):
            self.layers_deep.append(Layer(n_layers_len))
        # creating the output layer
        self.outputs = Layer(n_outputs)

        # connecting the input to the first deep layer
        for i in self.inputs.neurons:
            for j in self.layers_deep[0].neurons:
                i.next.append(j)
                j.prev.append(i)
                j.prev_weight.append(0.01 * np.random.randn())
        # connecting the deep layers together
        for i in range(n_layers-1):
            for neuron1 in self.layers_deep[i].neurons:
                for neuron2 in self.layers_deep[i+1].neurons:
                    neuron1.next.append(neuron2)
                    neuron2.prev.append(neuron1)
                    neuron2.prev_weight.append(0.01 * np.random.randn())

        # connecting the final deep layer to the output layer
        for neuron in self.layers_deep[n_layers-1].neurons:
            for outputNeuron in self.outputs.neurons:
                neuron.next.append(outputNeuron)
                outputNeuron.prev.append(neuron)
                outputNeuron.prev_weight.append(0.01 * np.random.randn())

    def forward(self, inputs)


def main():
    pass
    a = Layer(5)
    # for neuron in a.neurons:
    #     print(neuron.val)
    b = Network(3, 3, 10, 3)
    for neuron in b.inputs.neurons:
        neuron.val = 5
    for layer in b.layers_deep:
        layer.forward(1)
    for neuron in b.layers_deep[-1].neurons:
        print(neuron.val)


if __name__ == '__main__':
    pass
    # Network(3, 3, 10, 3)
    # print(np.random.randn())
    main()
