import numpy as np
import pandas as pd
import csv

from layer import Starter, Ender, Utils, Dense
from activation import *


class Network:
    def __init__(
        self,
        batch_size,
        learning_rate,
        layers,
        train_file_path="input/train.csv",
        normalization=Utils.stable_normalize,
        float_type="float32",
    ):
        self.layers = layers
        self.layers.insert(
            0,
            Starter(
                train_file_path=train_file_path,
                normalization=normalization,
                float_type=float_type,
            ),
        )
        self.layers.append(Ender())

        for i in range(len(self.layers)):
            self.layers[i].init(self.layers[i - 1], batch_size, learning_rate)

        # functions
        self.run = self.gradient_descent
        self.get_predictions = lambda a: np.argmax(a, 0)
        self.get_accuracy = (
            lambda predictions, expected: np.sum(predictions == expected)
            / expected.size
        )
        self.normalize = normalization

        # other class variables
        self.batch_size = batch_size
        self.float_type = float_type

    def forward(self):
        for i in range(len(self.layers)):
            self.layers[i].forward(self.layers[i - 1])

    def backward(self):
        for i in range(len(self.layers) - 1, -1, -1):
            self.layers[i].backward(
                self.layers[i - 1], self.layers[(i + 1) % len(self.layers)]
            )

    def update(self):
        for i in range(len(self.layers)):
            self.layers[i].update()

    def gradient_descent(
        self, epochs, report_accuracy=True, learning_rate_reduction=None
    ):
        iterations = int(np.ceil(self.layers[0]._x.shape[0] / self.batch_size))
        for epoch in range(1, epochs + 1):
            for iteration in range(iterations):
                self.forward()
                self.backward()
                self.update()

            # reduce learning rate
            if learning_rate_reduction != None:
                assert 0 < learning_rate_reduction < 1
                for layer in self.layers:
                    if type(layer) != Starter and type(layer) != Ender:
                        layer.alpha *= learning_rate_reduction

            # report accuracy
            if not epoch % report_accuracy:
                predictions = self.get_predictions(self.layers[-2].a)
                accuracy = self.get_accuracy(predictions, self.layers[-1].expected)
                #print(f"Epoch:\t{epoch}/{epochs}\nAccuracy:\t{accuracy}\n")
                yield (epoch, accuracy)

    def submit_csv(self, data_path, output_path):
        test_data = pd.read_csv(data_path).to_numpy()
        a = self.normalize(test_data.T).astype(self.float_type)
        del test_data

        for i in range(1, len(self.layers) - 1):
            z = self.layers[i].w.dot(a) + self.layers[i].b
            a = self.layers[i].act(z)

        predictions = self.get_predictions(a)

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ImageId", "Label"])
            writer.writerows(list(enumerate(predictions, 1)))
