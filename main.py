__author__ = "Yehya Albakri"

import numpy as np

# create neuron (Node)


class Node():
    def __init__(self, capacity=2, charge=0, leakage=0, prev=[], prev_weight=[], next=[]):
        self.capacity = capacity
        self.charge = charge
        self.leakage = leakage
        self.prev = prev
        self.prev_weight = prev_weight
        self.next = next


class Layer():
    pass

# class Network():
#     def __init__(self, n_inputs, n_layer1, n_outputs):
#         self.inputs = []
#         self.layer1 = []
#         for i in range(n_inputs):
#             self.inputs[i] = Node()
#         for i in range(n_layer1):
#             self.


def run(input1, input2, input3):
    I1 = Node()
    I2 = Node()
    I3 = Node()
    F1 = Node()
    layer1 = []
    for i in range(15):
        layer1.append(Node())
        layer1[i].prev.append(I1)
        layer1[i].prev.append(I2)
        layer1[i].prev.append(I3)
        I1.next.append(layer1[i])
        I2.next.append(layer1[i])
        I3.next.append(layer1[i])
        layer1[i].prev_weight = 0.1 * np.random.randn(3, 1)
        F1.prev.append(layer1[i])
        layer1[i].next.append(F1)
    return (I1, I2, I3, F1, layer1)
    # print(layer1)


def main():
    pass


if __name__ == '__main__':
    # print(Node().next)
    pass
run(1, 2, 3)
