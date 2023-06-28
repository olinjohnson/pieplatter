from typing import List
import matplotlib.pyplot as plt
import numpy as np

class InvalidShapesError(Exception):
    pass

class TrainingComparePlotter:
    def __init__(self, training_iters, num_epochs):
        self.fig = plt.figure(figsize=(13, 3 * int(np.ceil(training_iters / 3))))
        self.grid = plt.GridSpec(int(np.ceil(training_iters / 3)) + 1, 3, wspace=0.3, hspace=0.6)
        self.num_epochs = num_epochs
        self.training_iters = training_iters

    def plot_data(self, losses):
        # TODO: edit assertion to encompass training epochs
        try:
            assert len(losses) == self.training_iters
        except AssertionError:
            raise InvalidShapesError("Incompatible shapes: number of training iterations and number of losses do not "
                                     "match.")

        for i in range(0, self.training_iters):
            sp = plt.subplot(self.grid[i % 3, int(np.floor(i / 3))])
            sp.set_title(f'Iteration {i} loss')
            sp.set_ylim(0.0, 0.7)
            sp.plot([x for x in range(0, self.num_epochs)], [x for x in losses[i]])

        plt.show()

