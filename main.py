import numpy as np

class ActivationFunction:
    def __init__(self, activation_type):
        self.activation_type = activation_type
        self.input = None
        self.output = None

    def forward(self, input_data):
        self.input = input_data
        if self.activation_type == 'sigmoid':
            self.output = 1 / (1 + np.exp(-input_data))
        elif self.activation_type == 'relu':
            self.output = np.maximum(0, input_data)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_type}")

        return self.output

    def backward(self, gradient):
        if self.activation_type == 'sigmoid':
            grad_input = self.output * (1 - self.output) * gradient
        elif self.activation_type == 'relu':
            grad_input = (self.input > 0) * gradient
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_type}")

        return grad_input

class NeuralNetwork:
    def __init__(self):
        self.layers = []

    def add_layer(self, input_size, output_size, activation_type='sigmoid'):
        layer = {
            'weights': np.random.randn(input_size, output_size),
            'bias': np.zeros((1, output_size)),
            'activation': ActivationFunction(activation_type),
            'input': None,
            'output': None
        }
        self.layers.append(layer)

    def forward(self, input_data):
        current_input = input_data
        for layer in self.layers:
            layer['input'] = current_input
            layer['output'] = np.dot(current_input, layer['weights']) + layer['bias']
            current_input = layer['activation'].forward(layer['output'])
        return current_input

    def backward(self, loss_gradient, learning_rate):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            activation_gradient = layer['activation'].backward(loss_gradient)

            # Reshape input to ensure it's a 2D array before taking transpose
            reshaped_input = layer['input'].reshape(-1, 1)
            weights_gradient = np.dot(reshaped_input, activation_gradient)
            bias_gradient = np.sum(activation_gradient, axis=0, keepdims=True)

            loss_gradient = np.dot(activation_gradient, layer['weights'].T)

            # Update weights and bias
            layer['weights'] -= learning_rate * weights_gradient
            layer['bias'] -= learning_rate * bias_gradient

        return loss_gradient
# XOR input and corresponding output
xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([[0], [1], [1], [0]])

# Instantiate Neural Network
nn_xor = NeuralNetwork()

# Add layers with activation functions
nn_xor.add_layer(input_size=2, output_size=5, activation_type='relu')  # Hidden layer
# nn_xor.add_layer(input_size=5, output_size=3, activation_type='sigmoid')  # Hidden layer
nn_xor.add_layer(input_size=5, output_size=1, activation_type='sigmoid')  # Output layer

# Training parameters
epochs = 100000
learning_rate = 0.1

# Training loop
for epoch in range(epochs):
    total_loss = 0

    for i in range(len(xor_inputs)):
        # Forward pass
        output = nn_xor.forward(xor_inputs[i])

        # Compute loss (mean squared error)
        loss = 0.5 * np.sum((output - xor_outputs[i]) ** 2)
        total_loss += loss

        # Compute loss gradient
        loss_gradient = output - xor_outputs[i]

        # Backward pass with gradient descent
        nn_xor.backward(loss_gradient, learning_rate)

    # Print the average loss for every 1000 epochs
    if epoch % 1000 == 0:
        average_loss = total_loss / len(xor_inputs)
        print(f"Epoch {epoch}, Loss: {average_loss}")

# Test the trained network
print("\nTesting the trained network:")
for i in range(len(xor_inputs)):
    predicted_output = nn_xor.forward(xor_inputs[i])
    print(f"Input: {xor_inputs[i]}, Predicted Output: {predicted_output}")
