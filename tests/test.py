import numpy as np
import pieplatter as pie

net = pie.Network([
    pie.Layer(2, 3, activation=pie.ActivationTanh),
    pie.Layer(3, 1, activation=pie.ActivationSigmoid)
], pie.LossCCE)

inputs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
expected = np.array([[0], [1], [1], [0]])

loss, cache = net.forward_prop(inputs, expected)
print("LOSS: ", loss)
print("CACHE: \n", cache)
