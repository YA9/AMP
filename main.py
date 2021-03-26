__author__ = "Yehya Albakri"

import numpy as np

# create neuron (Node)


class Node():
    def __init__(self, prev=None, prev_weight=None, next=None):
        self.prev = prev
        self.prev_weight = prev_weight
        self.next = next


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
            self.neurons.add(newNode)


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
        # connecting the deep layers together
        for i in range(n_layers-1):
            for neuron1 in self.layers_deep[i].neurons:
                for neuron2 in self.layers_deep[i+1].neurons:
                    neuron1.next.append(neuron2)
                    neuron2.prev.append(neuron1)

        # connecting the final deep layer to the output layer
        for neuron in self.layers_deep[n_layers-1].neurons:
            for outputNeuron in self.outputs.neurons:
                neuron.next.append(outputNeuron)
                outputNeuron.prev.append(neuron)


# def run(input1, input2, input3):
#     I1 = Node()
#     I2 = Node()
#     I3 = Node()
#     F1 = Node()
#     layer1 = []
#     for i in range(15):
#         layer1.append(Node())
#         layer1[i].prev.append(I1)
#         layer1[i].prev.append(I2)
#         layer1[i].prev.append(I3)
#         I1.next.append(layer1[i])
#         I2.next.append(layer1[i])
#         I3.next.append(layer1[i])
#         layer1[i].prev_weight = 0.1 * np.random.randn(3, 1)
#         F1.prev.append(layer1[i])
#         layer1[i].next.append(F1)
#     return (I1, I2, I3, F1, layer1)
    # print(layer1)


def main():
    pass


if __name__ == '__main__':
    # print(Node().next)
    pass
    # print(Layer(5).neurons)
    Network(3, 3, 10, 3)
    # A = Node()
    # B = Node()
    # A.next.append(1)
    # print(B.next)
    # print(Network(3, 3, 10, 3).inputs.neurons[0].next)
