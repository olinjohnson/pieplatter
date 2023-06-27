from abc import ABC
import numpy as np
# TODO: add docstrings

class Activation(ABC):

    @staticmethod
    def forward(inputs):
        pass

    @staticmethod
    def backward(inputs):
        pass


class ActivationReLU(Activation):

    @staticmethod
    def forward(inputs):
        return np.maximum(inputs, 0)

    @staticmethod
    def backward(inputs):
        return np.greater(inputs, 0) * 1


class ActivationLeakyReLU(Activation):

    @staticmethod
    def forward(inputs):
        return np.maximum(inputs, 0.01 * inputs)

    @staticmethod
    def backward(inputs):
        return np.where(inputs < 0, 0.01, 1)


class ActivationELU(Activation):

    @staticmethod
    def forward(inputs):
        return np.maximum(inputs, np.exp(inputs) - 1)

    @staticmethod
    def backward(inputs):
        # TODO: implement derivative function
        pass


class ActivationSigmoid(Activation):

    @staticmethod
    def forward(inputs):
        return 1 / (np.exp(-inputs) + 1)

    @staticmethod
    def backward(inputs):
        return inputs * (1 - inputs)


class ActivationTanh(Activation):

    @staticmethod
    def forward(inputs):
        p = np.exp(inputs)
        n = np.exp(-inputs)
        return (p - n) / (p + n)

    @staticmethod
    def backward(inputs):
        return 1 - (inputs ** 2)


class ActivationSoftmax(Activation):

    @staticmethod
    def forward(inputs):
        # TODO: verify functionality
        i_exp = np.exp(inputs)
        s = np.sum(i_exp, axis=1)
        return np.divide(i_exp.T, s).T

    @staticmethod
    def backward(inputs):
        # TODO: implement derivative function
        pass
