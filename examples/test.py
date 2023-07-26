import numpy as np
import pieplatter as pie
import matplotlib.pyplot as plt

def test_xor():
    net = pie.Network([
        pie.Layer(2, 3, activation=pie.ActivationSigmoid),
        pie.Layer(3, 3, activation=pie.ActivationSigmoid),
        pie.Layer(3, 1, activation=pie.ActivationSigmoid)
    ], pie.LossMSE)

    inputs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    expected = np.array([[1], [0], [0], [0]])

    # loss, cache = net.forward_prop(inputs, expected)
    # print("LOSS BEFORE TRAINING: ", loss)
    # net.train(inputs, expected, 5000, 0.2)
    # loss, cache = net.forward_prop(inputs, expected)
    # print("LOSS AFTER TRAINING: ", loss)

    net.set_params(5000, 0.2)

    ruler = pie.TrainingMetrics(net)
    plot_machine = pie.MultiLinePlotter()
    mean_loss = ruler.measure_iterations(10, inputs, expected, plotter=plot_machine)
    print("MEAN LOSS: ", mean_loss)

test_xor()
