from typing import List
import numpy as np
import time
from pieplatter.src.network import Network
from pieplatter.src.metrics.plotting import Plotter


class TrainingMetrics:
    def __init__(self, net: Network):
        self.net = net

    def measure_iterations_loss(self, training_iters, input_data, expected, plotter: Plotter = None, logging=True):
        """
        Measures the average error of the network over multiple training iterations
        :param training_iters: the number of training iterations
        :param input_data: the training data
        :param expected: the training data labels
        :param logging: True to log progress, False otherwise
        :param plotter: The plotter to use for graphing the training data
        :return: the mean loss of the network after ``training_iters`` training iterations
        """
        losses = []
        for i in range(0, training_iters):
            self.net.reset()
            losses.append(self.net.train(input_data, expected))
            if logging:
                print("Finished training iteration ", i)
        if plotter:
            plotter.plot_losses(np.array(losses))
        return np.mean([x[-1] for x in losses])

    @staticmethod
    def time(callback, *args, **kwargs):
        start = time.time()
        callback(*args, **kwargs)
        end = time.time()
        return end - start
