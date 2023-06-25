import numpy as np
import numpy.random
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, inputs, outputs):
        self.weights = np.random.randn(outputs, inputs)
        self.biases = np.random.randn(outputs)

    def calc(self, inputs):
        return np.dot(inputs, self.weights.T) + self.biases

    @staticmethod
    def activation_relu(inputs):
        return np.maximum(inputs, 0)

    @staticmethod
    def activation_leaky_relu(inputs):
        return np.maximum(inputs, 0.01 * inputs)

    @staticmethod
    def activation_softmax(inputs):
        i_exp = np.exp(inputs)
        s = np.sum(i_exp, axis=1)
        return np.divide(i_exp.T, s).T

    @staticmethod
    def activation_sigmoid(inputs):
        return 1 / (np.exp(-inputs) + 1)

    @staticmethod
    def activation_tanh(inputs):
        numer = np.exp(inputs) - np.exp(-inputs)
        denom = np.exp(inputs) + np.exp(-inputs)
        return numer / denom

    @staticmethod
    def loss_mse(output, expected):
        squares = (output - expected) ** 2
        return np.sum(squares) / len(expected)

def derivative_sigmoid(x):
    return x * (1 - x)

def derivative_tanh(x):
    return 1 - (x ** 2)

def derivative_relu(x):
    return np.greater(x, 0) * 1

def derivative_leaky_relu(x):
    return np.where(x > 0, 1, 0.01)


inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Remember one hot? is this the correct input format?
expected = np.array([[0], [1], [1], [0]])

def forward_prop(model):
    z0 = model[0].calc(inputs)
    a0 = Layer.activation_tanh(z0)
    z1 = model[1].calc(a0)
    a1 = Layer.activation_sigmoid(z1)
    loss = Layer.loss_mse(a1, expected)
    # print("FEEDFORWARD OUTPUT: ", a1)
    cache = {"z0": z0, "a0": a0, "z1": z1, "a1": a1}
    return loss, cache


def back_prop(model, cache):

    dca1 = 2 * (cache["a1"] - expected)
    da1z1 = derivative_sigmoid(cache["a1"])
    dz1w1 = cache["a0"]

    dz1a0 = model[1].weights
    da0z0 = derivative_tanh(cache["a0"])
    dz0w0 = inputs

    dcb1 = np.mean(dca1 * da1z1, axis=0)
    model[1].biases -= (dcb1 * LEARNING_RATE)

    dcw1 = np.dot((dca1 * da1z1).T, dz1w1) / len(dca1 * da1z1)  # <- batch size
    model[1].weights -= (dcw1 * LEARNING_RATE)

    dcb0 = np.mean(dca1 * da1z1 * dz1a0 * da0z0, axis=0)
    model[0].biases -= (dcb0 * LEARNING_RATE)

    dca0 = np.dot((dca1 * da1z1), dz1a0) / len(dca1)
    dcw0 = np.dot((dca0 * da0z0).T, dz0w0) / len(dca0 * da0z0)
    model[0].weights -= (dcw0 * LEARNING_RATE)


def train_data(num_iters):
    print(f'Beginning {num_iters} training iterations...')
    losses = []
    differentials = []
    for x in range(0, num_iters):
        model = [
            Layer(2, 3),
            Layer(3, 1)
        ]
        losses.append([])
        l, c = forward_prop(model)
        il = l
        for i in range(0, NUM_EPOCHS):
            l, c = forward_prop(model)
            losses[x].append(l)
            back_prop(model, c)
        l, c = forward_prop(model)
        differentials.append(il - l)
        print(f'Finished training set {x}...Continuing')
    print(f'Finished {num_iters} training iterations...Plotting data\n')
    return losses, differentials

LEARNING_RATE = 0.2
NUM_EPOCHS = 20000
NUM_TRAIN_ITERS = 10

loss, diff = train_data(NUM_TRAIN_ITERS)

print("Loss differentials:")
for i in range(0, len(diff)):
    print(f'Iteration {i} differential: {diff[i]}')

fig = plt.figure(figsize=(9, 16))
grid = plt.GridSpec(7, 2, wspace=0.3, hspace=0.6)
for i in range(0, 5):
    for x in range(0, 2):
        sp = plt.subplot(grid[i, x])
        sp.set_title(f'Iteration {i*2+x} loss')
        sp.set_ylim(0.0, 0.7)
        sp.plot([x for x in range(0, len(loss[i*2+x]))], [x for x in loss[i*2+x]])

sp = plt.subplot(grid[5:7, :2])
# sp.bar([x for x in range(0, NUM_TRAIN_ITERS)], diff)
sp.bar([x for x in range(0, NUM_TRAIN_ITERS)], [x[-1] for x in loss])
sp.set_title("Final Losses")
sp.set_ylim(0, 0.5)
plt.show()
print("\nFinished plotting data")
