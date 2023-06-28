import numpy as np
import pieplatter as pie
import time

def test_xor():
    net = pie.Network([
        pie.Layer(2, 2, activation=pie.ActivationTanh),
        pie.Layer(2, 3, activation=pie.ActivationTanh),
        pie.Layer(3, 1, activation=pie.ActivationSigmoid)
    ], pie.LossMSE)

    inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    expected = np.array([[0], [1], [1], [0]])

    loss, cache = net.forward_prop(inputs, expected)
    print("LOSS: ", loss)

    # start = time.time()

    losses = []
    for i in range(0, 4):
        net.reset()
        losses.append(net.train(inputs, expected, 5000, 0.2))
        print("Finished iteration ", i)

    # end = time.time()

    loss, cache = net.forward_prop(inputs, expected)
    print("LOSS: ", loss)
    # print("TRAINING TIME: ", (end - start))

    plotter = pie.TrainingComparePlotter(4, 5000)
    plotter.plot_data(losses)

test_xor()

# 0.923145055770874
