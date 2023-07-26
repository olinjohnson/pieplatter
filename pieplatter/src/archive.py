# TIME PLOTTING CODE:
# def plot_times(self, data, times):
#     if len(data.shape) != 2 or len(times.shape) != 1:
#         raise InvalidShapesError("Invalid data shape for 2-dimensional plot")
#
#     training_iters = len(data)
#     num_epochs = len(data[0])
#
#     plt.bar([x for x in range(0, training_iters)], [x for x in times], width=0.3)
#     plt.plot([x for x in range(0, training_iters)], [x[-1] for x in data], color="orange")
#     plt.xticks(range(0, training_iters))
#     plt.title("Network training times by iteration")
#     plt.xlabel("Training iteration number")
#     plt.ylabel("Training time")
#     plt.show()


# OUTPUT LAYER:
# class OutputLayer(Layer):
#     def __init__(self, num_inputs: int, num_outputs: int, activation: Type[Activation], loss: Type[Loss]):
#         super().__init__(num_inputs, num_outputs, activation)
#         self.loss = loss
#
#     def forward_loss(self, inputs, expected):
#         a, cache = super().forward(inputs)
#         return self.loss.forward(inputs, expected), cache
#
