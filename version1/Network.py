import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import csv

class Network:
    # input layer -> hiddenlayer (1) -> output layer
    def __init__(self, train_file_path, hidden_layer_size=256, batch_size=500, iterations=500, learning_rate=0.1):
        np.random.seed(1)

        # read files/data
        print("Starting to read train data...")
        self.train = pd.read_csv(train_file_path).to_numpy()
        print("Finished reading train data")

        # initiate constant variables
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate

        # initiate random weights and biases
        self.w1 = np.random.rand(hidden_layer_size, 784) - 0.5
        self.b1 = np.random.rand(hidden_layer_size, 1) - 0.5

        self.w2 = np.random.rand(10, hidden_layer_size) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

        # initiate layers
        self.z1 = None
        self.a1 = None  # hidden layer

        self.z2 = None
        self.a2 = None  # output layer

        # initiate back prop
        self.dz1 = None  # derivative of hidden layer
        self.dw1 = None
        self.db1 = None

        self.dz2 = None  # derivative of output layer
        self.dw2 = None
        self.db2 = None

    ### Mathematical functions
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        self.sigmoid(x) * (1 - self.sigmoid(x))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_deriv(self, x):
        return x > 0

    def softmax(self, x):
        return np.exp(x) / sum(np.exp(x))

    ### other helper functions
    def get_batch_rand(self):
        indices = np.random.choice(self.train.shape[0], self.batch_size, replace=False)
        choices = self.train[indices]

        inputs = choices.T[1:] * 0.99 / 255 + 0.01
        inputs = inputs.astype("float32")
        expected = choices.T[0]

        return inputs, expected

    def one_hot(self, x):
        one_hot_x = np.zeros((x.size, 10))
        one_hot_x[np.arange(x.size), x] = 1
        return one_hot_x.T

    ### training functions
    def forward_prop(self, inputs):
        # input layer -> hidden layer
        self.z1 = self.w1.dot(inputs) + self.b1
        self.a1 = self.relu(self.z1)

        # hidden layer -> output layer
        self.z2 = self.w2.dot(self.a1) + self.b2
        self.a2 = self.softmax(self.z2)

    def back_prop(self, inputs, expected):
        # output layer -> hidden layer
        one_hot_y = self.one_hot(expected)
        self.dz2 = self.a2 - one_hot_y
        self.dw2 = self.dz2.dot(self.a1.T) / self.batch_size
        self.db2 = np.sum(self.dz2) / self.batch_size

        # hidden layer -> input layer
        self.dz1 = self.w2.T.dot(self.dz2) * self.relu_deriv(self.z1)
        self.dw1 = self.dz1.dot(inputs.T) / self.batch_size
        self.db1 = np.sum(self.dz1) / self.batch_size

        ## update weights and biases
        self.w1 = self.w1 - self.learning_rate * self.dw1
        self.b1 = self.b1 - self.learning_rate * self.db1
        self.w2 = self.w2 - self.learning_rate * self.dw2
        self.b2 = self.b2 - self.learning_rate * self.db2

    def gradient_descent(self, report_accuracy=10):
        for iteration in range(1, self.iterations + 1):
            inputs, expected = self.get_batch_rand()
            self.forward_prop(inputs)
            self.back_prop(inputs, expected)
            if report_accuracy > 0 and not iteration % report_accuracy:
                print("Iteration\t:", iteration)
                predictions = self.get_predictions(self.a2)
                self.get_accuracy(predictions, expected)
                print()

    # training accuracy predictions
    def get_predictions(self, a2):
        return np.argmax(a2, 0)

    def get_accuracy(self, predictions, expected):
        #print("Predictions\t:", predictions)
        #print("Expected\t:", expected)
        print("Accuracy\t:", np.sum(predictions == expected) / expected.size)

    # predictions with random values
    def predict_rand(self, show_image=False):
        index = np.random.choice(self.train.shape[0], 1, replace=False)
        choice = self.train[index]

        inputs = choice.T[1:] * 0.99 / 255 + 0.01
        expected = choice.T[0]

        # forward propagation without affecting actual weights or values
        z1 = self.w1.dot(inputs) + self.b1
        a1 = self.relu(z1)

        z2 = self.w2.dot(a1) + self.b2
        a2 = self.softmax(z2)

        # get predictions
        prediction = self.get_predictions(a2)

        # display data
        if show_image:
            choice = choice.T[1:].reshape((28, 28))
            plt.title(f"Expected: {int(expected)} Predicted: {int(prediction)} Confidence: {round(max(a2.T[0]) * 100, 2)}%\nVerdict: {'Correct' if int(expected) == int(prediction) else 'Incorrect'}")
            plt.gray()
            plt.imshow(choice, interpolation="nearest")
            plt.show()
        else:
            print(f'Expected\t: {int(expected)}\nPredicted\t: {int(prediction)}\nConfidence\t: {round(max(a2.T[0]) * 100, 2)}%\n')

    # predictions with test data
    def predict_sub(self, test_file_path, output_file_path):
        # read input file
        print("Starting to read test data...")
        test = pd.read_csv(test_file_path).to_numpy()
        print("Finished reading test data")

        inputs = test.T * 0.99 / 255 + 0.01
        inputs = inputs.astype("float32")

        # forward propagation without affecting actual weights or values
        z1 = self.w1.dot(inputs) + self.b1
        a1 = self.relu(z1)

        z2 = self.w2.dot(a1) + self.b2
        a2 = self.softmax(z2)

        predictions = self.get_predictions(a2)

        # write to output file
        rows = list(enumerate(predictions, 1))
        with open(output_file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ImageId", "Label"])
            writer.writerows(rows)

    # record weights
    def record_weights(self, w1, b1, w2, b2):
        np.savetxt(w1, self.w1, delimiter=",", newline="], [", header="[[", footer="]]")
        np.savetxt(b1, self.b1, delimiter=",", newline="], [", header="[[", footer="]]")
        np.savetxt(w2, self.w2, delimiter=",", newline="], [", header="[[", footer="]]")
        np.savetxt(b2, self.b2, delimiter=",", newline="], [", header="[[", footer="]]")