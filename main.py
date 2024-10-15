import numpy as np

# Initialize weights and biases
input_neurons = 36
hidden_neurons_1 = 10  # Increased first hidden layer neurons
hidden_neurons_2 = 5  # Added a second hidden layer
output_neurons = 2
learning_rate = 0.01  # Reduced learning rate
epochs = 10000

# Both teams' stats
training_data = np.array([
    [19.6, 2, 1.5, 44.5, 0.5, 0.5, 15.2, 0.7, 0.5, 40.7, 0.3, 0.5, 80, 50, 43.5, 5, 5, 9, 18.4, 1.2, 0.6, 60.7, 0.8,
     1.1, 18.8, 2, 1.2, 47.9, 0.6, 0.8, 72.2, 55, 46.8, 10, 42, -4],
    [21.1, 3.2, 2.3, 48, 0.3, 0.5, 16.3, 1.5, 0.9, 45.4, 0.5, 0.8, 85, 49, 45.4, 15, 4, 9, 21.5, 1.9, 1.3, 54.9, 0.3, 1,
     19.9, 2.2, 1.2, 63.3, 0.8, 0.2, 50, 50, 49, 10, 24, -2],
    [26.8, 3.5, 1.8, 59.2, 0.2, 0.6, 21.8, 1.6, 1.3, 38.2, 0.6, 0.9, 95.2, 31, 42.3, 27, 4, 9, 18.7, 2.1, 1.8, 63.9,
     0.5, 0.8, 29.2, 2.1, 0.8, 46.3, 0.8, 0.8, 63.2, 59, 48.4, 13, 23, 9],
    [19.8, 2.8, 2.4, 54.4, 0.4, 0.8, 26.1, 2, 1.3, 51.4, 1.5, 0.9, 71.4, 27, 45, 15, 2, 6, 18.9, 1.6, 0.3, 63, 1.1, 1.7,
     21.6, 1.1, 2.9, 50.4, 0.9, 0.7, 75, 37, 43.6, 30, 46, -3]
])

# Normalization
min_vals = training_data.min(axis=0)
max_vals = training_data.max(axis=0)
normalized_data = (training_data - min_vals) / (max_vals - min_vals)

# Target values
target_data = np.array([
    [38, 7],
    [38, 3],
    [49, 20],
    [56, 7]
])

# Initialize weights
np.random.seed(0)  # Seed
weights_input_hidden1 = np.random.rand(input_neurons, hidden_neurons_1)
weights_hidden1_hidden2 = np.random.rand(hidden_neurons_1, hidden_neurons_2)
weights_hidden2_output = np.random.rand(hidden_neurons_2, output_neurons)


# Activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative
def sigmoid_derivative(x):
    return x * (1 - x)


# Training
for epoch in range(epochs):
    # Forward propagation
    hidden_layer1_input = np.dot(normalized_data, weights_input_hidden1)
    hidden_layer1_output = sigmoid(hidden_layer1_input)

    hidden_layer2_input = np.dot(hidden_layer1_output, weights_hidden1_hidden2)
    hidden_layer2_output = sigmoid(hidden_layer2_input)

    # Regression test
    predicted_output = np.dot(hidden_layer2_output, weights_hidden2_output)

    # Calculate error (MSE)
    error = np.mean((target_data - predicted_output) ** 2)

    # Backpropagation
    output_error = target_data - predicted_output
    output_delta = output_error  # Linear derivative is 1

    hidden_layer2_error = output_delta.dot(weights_hidden2_output.T)
    hidden_layer2_delta = hidden_layer2_error * sigmoid_derivative(hidden_layer2_output)

    hidden_layer1_error = hidden_layer2_delta.dot(weights_hidden1_hidden2.T)
    hidden_layer1_delta = hidden_layer1_error * sigmoid_derivative(hidden_layer1_output)

    # Update weights
    weights_hidden2_output += hidden_layer2_output.T.dot(output_delta) * learning_rate
    weights_hidden1_hidden2 += hidden_layer1_output.T.dot(hidden_layer2_delta) * learning_rate
    weights_input_hidden1 += normalized_data.T.dot(hidden_layer1_delta) * learning_rate

    # Print
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {error}")

# Final predictions
predicted_output = np.dot(sigmoid(np.dot(sigmoid(np.dot(normalized_data, weights_input_hidden1)),
                                         weights_hidden1_hidden2)),
                          weights_hidden2_output)
print("Predictions:")
for i, (inputs, pred, actual) in enumerate(zip(normalized_data, predicted_output, target_data)):
    print(f"Input: {inputs} - Pred.: {pred} - Actual: {actual}")
