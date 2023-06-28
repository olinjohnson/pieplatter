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

    def reset(self):
        self.weights = np.random.randn(self.weights.shape[0], self.weights.shape[1])
        self.biases = np.random.randn(self.biases.shape[0])

    def calc(self, inputs):
        return np.dot(inputs, self.weights.T) + self.biases

    def activate(self, inputs):
        return self.activation.forward(inputs)

    def forward(self, inputs) -> Tuple[np.ndarray, List]:
        z = self.calc(inputs)
        a = self.activate(z)
        return a, [z, a]

    def backward(self, chain: np.ndarray, prev_act: np.ndarray, act: np.ndarray) -> List[np.ndarray]:
        """
        :param chain: a product of all the partial derivates up to this point
        :param prev_act: the cached outputs of the previous layer
        :param act: the cached outputs of the current layer
        :return: [np.ndarray, np.ndarray, np.ndarray]
        returns a list containing the derivative of z,
        a list of updated weights, and a list of updated biases.
        """
        a = self.activation.backward(act) * chain
        weight_update = np.dot(a.T, prev_act) / len(a)
        cont_chain = np.dot(a, self.weights)
        return [cont_chain, weight_update, np.mean(a, axis=0)]




# class OutputLayer(Layer):
#     def __init__(self, num_inputs: int, num_outputs: int, activation: Type[Activation], loss: Type[Loss]):
#         super().__init__(num_inputs, num_outputs, activation)
#         self.loss = loss
#
#     def forward_loss(self, inputs, expected):
#         a, cache = super().forward(inputs)
#         return self.loss.forward(inputs, expected), cache

