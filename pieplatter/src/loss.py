import numpy as np

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
    def forward(inputs, expected) -> np.ndarray:
        # TODO: make sure inputs and expected satisfy the
        #  precondition, i.e., have the same shape
        #
        # Only usable on multi-classification
        # problems; faulty otherwise
        return -np.sum(inputs * np.log(expected), axis=1)

    @staticmethod
    def backward(inputs, expected):
        # TODO: implement derivative function
        pass


class LossBCE(Loss):
    """
    Binary Cross Entropy loss function
    """
    @staticmethod
    def forward(inputs, expected) -> np.ndarray:
        # TODO: implement loss function
        pass

    @staticmethod
    def backward(inputs, expected):
        # TODO: implement derivative function
        pass
