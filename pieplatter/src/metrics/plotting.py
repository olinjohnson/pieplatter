import random

import matplotlib.pyplot as plt
import numpy as np
from pieplatter.src.utils import InvalidShapesError

class Plotter:
    def plot_losses(self, data: np.ndarray):
        pass

class MultiLinePlotter(Plotter):
    def plot_losses(self, data):
        if len(data.shape) != 2:
            raise InvalidShapesError("Invalid data shape for 2-dimensional plot")

        training_iters = len(data)
        num_epochs = len(data[0])

        fig = plt.figure(figsize=(10, 10))
        grid = plt.GridSpec(4, 1, hspace=1)
        sp_graph = plt.subplot(grid[:-1, 0])
        sp_graph.set_title("Network losses over time (per training iteration)")
        sp_graph.set_xlabel("Epoch")
        sp_graph.set_ylabel("Network loss")
        for i in range(0, training_iters):
            sp_graph.plot([x for x in range(0, num_epochs)], [x for x in data[i]])

        sp_bar = plt.subplot(grid[-1, 0])
        sp_bar.bar([x for x in range(0, training_iters)], [x[-1] for x in data])
        sp_bar.set_xticks([x for x in range(0, training_iters)])
        sp_bar.set_title("Final network losses by training iteration")
        sp_bar.set_xlabel("Training iteration number")
        sp_bar.set_ylabel("Network loss")
        sp_bar.set_ylim(0.0, 0.5)

        plt.show()

class MultiGraphPlotter(Plotter):
    def plot_losses(self, data):
        if len(data.shape) != 2:
            raise InvalidShapesError("Invalid data shape for 2-dimensional plot")

        training_iters = len(data)
        num_epochs = len(data[0])

        fig = plt.figure(figsize=(13, 3 * int(np.ceil(training_iters / 3))))
        grid = plt.GridSpec(int(np.ceil(training_iters / 3)) + 1, 3, wspace=0.3, hspace=0.6)

        colors = ['#' + ''.join([random.choice("0123456789abcdef") for x in range(0, 6)]) for x in range(0, training_iters)]

        for i in range(0, training_iters):
            sp = plt.subplot(grid[int(np.floor(i / 3)), i % 3])
            sp.set_title(f'Iteration {i} loss')
            sp.set_ylim(0.0, 0.7)
            sp.plot([x for x in range(0, num_epochs)], [x for x in data[i]], color=colors[i])

        sp_bar = plt.subplot(grid[-1, :3])
        sp_bar.bar([x for x in range(0, training_iters)], [x[-1] for x in data], color=colors)
        sp_bar.set_xticks([x for x in range(0, training_iters)])
        sp_bar.set_title("Final network losses by training iteration")
        sp_bar.set_xlabel("Training iteration number")
        sp_bar.set_ylabel("Network loss")
        sp_bar.set_ylim(0.0, 0.5)
        plt.show()
