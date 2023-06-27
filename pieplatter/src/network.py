from typing import List, Tuple, Type
import numpy as np
from pieplatter.src.layers import Layer
from pieplatter.src.loss import Loss


class Network:
    def __init__(self, model: List[Layer], loss: Type[Loss]):
        self.model = model
        self.loss = loss

    def forward_prop(self, inputs, expected) -> Tuple[np.ndarray, List]:
        cache = []
        curr_val = inputs
        for layer in self.model:
            curr_val, lay_cache = layer.forward(curr_val)
            cache.append(lay_cache)
        prop_loss = self.loss.forward(cache[-1][-1], expected)
        return np.mean(prop_loss), cache

    def train(self, training_data, training_labels, num_epochs, learning_rate):
        for x in range(0, num_epochs):
            cost, cache = self.forward_prop(training_data, training_labels)
            chain = self.loss.backward(cache[-1][-1], training_labels)
            for i in range(len(self.model) - 1, 0, -1):
                curr_layer = self.model[i]
                chain, weight_up, bias_up = curr_layer.backward(chain, cache[i - 1][-1], cache[i][-1])
                curr_layer.weights -= (weight_up * learning_rate)
                curr_layer.biases -= (bias_up * learning_rate)
            curr_layer = self.model[0]
            chain, weight_up, bias_up = curr_layer.backward(chain, training_data, cache[0][-1])
            curr_layer.weights -= (weight_up * learning_rate)
            curr_layer.biases -= (bias_up * learning_rate)
