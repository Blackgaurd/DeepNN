import activation

import numpy as np
import pandas as pd

np.random.seed(1)


class Utils:
    def __init__(self):
        pass

    @staticmethod
    def one_hot(y):
        """to categorical"""
        one_hot_y = np.zeros((y.size, 10))
        one_hot_y[np.arange(y.size), y] = 1
        return one_hot_y.T

    @staticmethod
    def normalize(x):
        return x / 255

    @staticmethod
    def stable_normalize(x):
        return x * 0.99 / 255 + 0.01

    @staticmethod
    def seed(val):
        np.random.seed(val)


class Dense:
    def __init__(self, size, activation):
        self.size = size
        self.act = activation.forward
        self.deriv = activation.backward
        (
            self.w,
            self.b,
            self.a,
            self.z,
            self.dw,
            self.db,
            self.dz,
            self.batch_size,
            self.alpha,
        ) = (None for i in range(9))

    def init(self, prev, batch_size, learning_rate):
        self.w = np.random.rand(self.size, prev.size) - 0.5
        self.b = np.random.rand(self.size, 1) - 0.5
        self.batch_size = prev.batch_size
        self.alpha = learning_rate

    def forward(self, prev):
        self.z = self.w.dot(prev.a) + self.b
        self.a = self.act(self.z)

    def backward(self, prev, post):
        self.dz = self.deriv(self, post)
        self.dw = self.dz.dot(prev.a.T) / self.batch_size
        self.db = np.sum(self.dz) / self.batch_size

    def update(self):
        self.w = self.w - self.dw * self.alpha
        self.b = self.b - self.db * self.alpha


class Starter:
    def __init__(self, train_file_path, normalization, float_type):
        train_data = pd.read_csv(train_file_path)
        self._y = train_data["label"].to_numpy()

        self._x = train_data.loc[:, train_data.columns != "label"].to_numpy()
        self._x = normalization(self._x).astype(float_type)

        self.size = self._x.shape[1]

        self.a, self.batch_size, self.indices = None, None, None

    def init(self, prev, batch_size, learning_rate):
        self.batch_size = batch_size
        prev._y = self._y.T
        del self._y  # (save on memory)

    def forward(self, prev):
        self.indices = np.random.choice(
            self._x.shape[0], self.batch_size, replace=False
        )
        self.a = self._x[self.indices].T

    def backward(self, *args, **kwargs):
        pass

    def update(self):
        pass


class Ender:
    def __init__(self):
        self.a, self._y, self.expected, self._one_hot_y = None, None, None, None

    def init(self, *args, **kwargs):
        self._one_hot_y = Utils.one_hot(self._y).T

    def forward(self, *args, **kwargs):
        pass

    def backward(self, prev, post):
        self.expected = self._y[post.indices]
        self.a = self._one_hot_y[post.indices].T

    def update(self):
        pass
