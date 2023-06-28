import numpy as np
from pieplatter import ActivationReLU

def test_relu():
    test_arr = np.array([[-1, 4, -3], [2, 2, -5]])
    expected_arr = np.array([[0, 4, 0], [2, 2, 0]])
    assert ActivationReLU.forward(test_arr) == expected_arr
    expected_grad_arr = np.array([[0, 1, 0], [1, 1, 0]])
    assert ActivationReLU.backward(expected_arr) == expected_grad_arr

# TODO: add more tests
