import numpy as np

from pieplatter.src.utils import not_implemented, not_implemented_correctly, not_reviewed


class Loss:

    @staticmethod
    def forward(inputs, expected) -> np.ndarray:
        pass

    @staticmethod
    def backward(inputs, expected):
        pass


class LossMSE(Loss):
    """
    Mean Squared Error loss function
    """

    @staticmethod
    @not_reviewed
    def forward(inputs, expected) -> np.ndarray:
        # TODO: make sure inputs and expected
        #  satisfy the precondition
        squares = (inputs - expected) ** 2
        return np.mean(squares, axis=1)

    @staticmethod
    def backward(inputs, expected):
        return 2 * (inputs - expected)


class LossCCE(Loss):
    """
    Categorical Cross Entropy loss function
    """
    @staticmethod
    @not_implemented_correctly
    def forward(inputs, expected) -> np.ndarray:
        # TODO: make sure inputs and expected satisfy the
        #  precondition, i.e., have the same shape
        #
        # Only usable on multi-classification
        # problems; faulty otherwise
        return -np.sum(inputs * np.log(expected), axis=1)

    @staticmethod
    @not_implemented
    def backward(inputs, expected):
        # TODO: implement derivative function
        pass


class LossBCE(Loss):
    """
    Binary Cross Entropy loss function
    """
    @staticmethod
    @not_implemented
    def forward(inputs, expected) -> np.ndarray:
        # TODO: implement loss function
        pass

    @staticmethod
    @not_implemented
    def backward(inputs, expected):
        # TODO: implement derivative function
        pass
