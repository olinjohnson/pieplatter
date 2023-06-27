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
        # TODO: verify that network last layer is an instance of OutputLayer

        output_layer = self.model[-1]

        for i in range(len(self.model) - 1, 0, -1):
            curr_layer = self.model[i - 1]


