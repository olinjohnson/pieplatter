import numpy as np
from typing import Tuple, List, Type
from pieplatter.src.activations import Activation
from pieplatter.src.loss import Loss
# TODO: add docstrings
# TODO: add conv and reccurent layers


class Layer:
    def __init__(self, num_inputs: int, num_outputs: int, activation: Type[Activation]):
        self.weights = np.random.randn(num_outputs, num_inputs)
        self.biases = np.random.randn(num_outputs)
        self.activation = activation

    def calc(self, inputs):
        return np.dot(inputs, self.weights.T) + self.biases

    def activate(self, inputs):
        return self.activation.forward(inputs)

    def forward(self, inputs) -> Tuple[np.ndarray, List]:
        z = self.calc(inputs)
        a = self.activate(z)
        return a, [z, a]


# class OutputLayer(Layer):
#     def __init__(self, num_inputs: int, num_outputs: int, activation: Type[Activation], loss: Type[Loss]):
#         super().__init__(num_inputs, num_outputs, activation)
#         self.loss = loss
#
#     def forward_loss(self, inputs, expected):
#         a, cache = super().forward(inputs)
#         return self.loss.forward(inputs, expected), cache

