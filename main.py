import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(1)

print("Reading training data...")
train_data = pd.read_csv("train.csv").to_numpy()
print("Finished reading training data.")
#print("Reading test data...")
#test_data = pd.read_csv("test.csv").to_numpy()
#print("Finished reading test data")

train_data = train_data[:1000]

input_layer = train_data.T[1:] / 255 # (784, 100)
expected = train_data.T[0] # (100, 1)

n = input_layer.shape[1]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def one_hot(x):
    one_hot_x = np.zeros((x.size, x.max() + 1))
    one_hot_x[np.arange(x.size), x] = 1
    return one_hot_x

def softmax(x):
    return np.exp(x) / sum(np.exp(x))

input_layer_size, hidden_layer_size, output_layer_size = input_layer.shape[0], 256, 10

w_hidden = np.random.rand(input_layer_size, hidden_layer_size)
w_output = np.random.rand(hidden_layer_size, output_layer_size)

b_hidden = np.random.rand(input_layer.shape[1], hidden_layer_size)
b_output = np.random.rand(1, output_layer_size)

# expected values in one hot format
y_one_hot = one_hot(expected)

def forward_prop():
    # input layer -> hidden layer
    z_hidden = input_layer.T.dot(w_hidden) + b_hidden

    # run through activation to get hidden layer
    a_hidden = sigmoid(z_hidden)

    # hidden layer -> output layer
    z_output = a_hidden.dot(w_output) + b_output

    # run through another activation to get output layer
    # use softmax instead of sigmoid because it provides
    # a better statistical approximation
    a_output = softmax(z_output)

    return z_hidden, a_hidden, z_output, a_output

def back_prop(z_hidden, a_hidden, z_output, a_output):
    # output layer -> hidden layer
    # calculate cost function: (implement mse later?)
    dz_output = a_output - y_one_hot
    # derivative of weight using error:
    dw_output = 1 / n * dz_output.T.dot(a_hidden)
    # derivative of output biases
    db_output = 1 / n * np.sum(dz_output)

    # hidden layer -> input layer
    # repeat steps from above
    # cost:
    dz_hidden = w_output.dot(dz_output.T) * sigmoid_deriv(z_hidden).T
    # derivative of weight:
    dw_hidden = 1 / n * dz_hidden.dot(input_layer.T)
    # derivative of hidden biases
    db_hidden = 1 / n * np.sum(dz_hidden)

    return dw_hidden, db_hidden, dw_output, db_output

def update(dw_hidden, db_hidden, dw_output, db_output, alpha):
    # update weights and biases of each layer
    # alpha = learning rate
    # the lower alpha is the more accuracy there is
    # at the comprimise of speed

    global w_hidden, b_hidden, w_output, b_output

    w_hidden = w_hidden - alpha * dw_hidden.T
    b_hidden = b_hidden - alpha * db_hidden

    w_output = w_output - alpha * dw_output.T
    b_output = b_output - alpha * db_output

def gradient_descent(iterations, alpha):
    for iteration in range(1, iterations+1):
        z_hidden, a_hidden, z_output, a_output = forward_prop()
        dw_hidden, db_hidden, dw_output, db_output = back_prop(z_hidden, a_hidden, z_output, a_output)
        update(dw_hidden, db_hidden, dw_output, db_output, alpha)

        if iteration % 10 == 0:
            print(f"Iteration: {iteration}")


if __name__ == "__main__":
    gradient_descent(100, 0.1)
    print("done")