import numpy as np
import pieplatter as pie
import matplotlib.pyplot as plt

def test_xor():
    net = pie.Network([
        pie.Layer(2, 3, activation=pie.ActivationTanh),
        pie.Layer(3, 3, activation=pie.ActivationTanh),
        pie.Layer(3, 1, activation=pie.ActivationSigmoid)
    ], pie.LossMSE)

    inputs = np.array([[1, 1], [0, 0], [1, 0], [0, 1]])
    expected = np.array([[0], [0], [1], [1]])

    net.set_params(1000, 0.1)
    t = pie.TrainingMetrics.time(net.train, inputs, expected)
    print("TIME: ", t)

test_xor()
