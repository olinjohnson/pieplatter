from typing import List, Tuple, Type, Any
import numpy as np
from pieplatter.src.layers import Layer
from pieplatter.src.loss import Loss
from pieplatter.src.utils import ParametersNotDefinedError


class InvalidInputException(Exception):
    pass

class Network:
    def __init__(self, model: List[Layer], loss: Type[Loss]):
        self.model = model
        self.loss = loss
        self.num_epochs = None
        self.learning_rate = None

    def reset(self):
        for layer in self.model:
            layer.reset()

    def set_params(self, num_epochs, learning_rate):
        """
        # TODO: add optimizers
        Set the training parameters for the network
        :param num_epochs: The number of epochs per training iteration
        :param learning_rate: The learning rate of the network
        :return:
        """
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

    def forward_prop(self, inputs, expected) -> Tuple[np.ndarray, List]:
        cache = []
        curr_val = inputs
        for layer in self.model:
            curr_val, lay_cache = layer.forward(curr_val)
            cache.append(lay_cache)
        prop_loss = self.loss.forward(cache[-1][-1], expected)
        return np.mean(prop_loss), cache

    def train(self, training_data, training_labels) -> List[Any]:
        """
        # TODO: fix function return type
        :param training_data: the training data
        :param training_labels: the labels for the training data
        :return: a list of loss values - one for each training epoch
        """
        if not self.num_epochs or not self.learning_rate:
            raise ParametersNotDefinedError("Network parameters have not been initialized")

        losses = []
        for x in range(0, self.num_epochs):
            cost, cache = self.forward_prop(training_data, training_labels)
            losses.append(cost)
            chain = self.loss.backward(cache[-1][-1], training_labels)
            for i in range(len(self.model) - 1, 0, -1):
                curr_layer = self.model[i]
                chain, weight_up, bias_up = curr_layer.backward(chain, cache[i - 1][-1], cache[i][-1])
                curr_layer.weights -= (weight_up * self.learning_rate)
                curr_layer.biases -= (bias_up * self.learning_rate)
            curr_layer = self.model[0]
            chain, weight_up, bias_up = curr_layer.backward(chain, training_data, cache[0][-1])
            curr_layer.weights -= (weight_up * self.learning_rate)
            curr_layer.biases -= (bias_up * self.learning_rate)
        return losses

