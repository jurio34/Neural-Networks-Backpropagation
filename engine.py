import numpy as np

def sigmoid(x):
    #activation function
    return 1 / (1 + np.exp(-x))
def sigmoid_derivative(x):
    #derivative of the sigmoid function
    return x * (1 - x)
def mean_squared_error(y_true, y_pred):
    # Mean Squared Error loss function
    return np.mean(np.power(y_true - y_pred, 2))
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.random.rand(hidden_size)
        self.bias_output = np.random.rand(output_size)

    def forward(self, x):
        # Forward pass through the network
        self.hidden_layer_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = sigmoid(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output = sigmoid(self.output_layer_input)
        return self.output
    def backward(self, x, y, output, learning_rate):
        # Backward pass through the network
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)

        hidden_layer_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(self.hidden_layer_output)

        # Update weights and biases
        self.weights_hidden_output += self.hidden_layer_output.T.dot(output_delta) * learning_rate
        self.bias_output += np.sum(output_delta, axis=0) * learning_rate
        self.weights_input_hidden += x.T.dot(hidden_layer_delta) * learning_rate
        self.bias_hidden += np.sum(hidden_layer_delta, axis=0) * learning_rate
