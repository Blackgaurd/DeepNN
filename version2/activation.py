import numpy as np

# Activation.forward() --> activation
# Activation.backward() --> derivative


class Relu:
    @staticmethod
    def forward(x):
        return np.maximum(0, x)

    @staticmethod
    def backward(cur, post):
        relu_d = lambda x: x > 0
        return post.w.T.dot(post.dz) * relu_d(cur.z)


class Sigmoid:
    @staticmethod
    def forward(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def backward(cur, post):
        sigmoid_d = lambda x: Sigmoid.forward(x) * (1 - Sigmoid.forward(x))
        return post.w.T.dot(post.dz) * sigmoid_d(cur.z)


class Tanh:
    @staticmethod
    def forward(x):
        return 2 * Sigmoid.forward(2 * x) - 1

    @staticmethod
    def backward(cur, post):
        tanh_d = lambda x: 1 - Tanh.forward(x) ** 2
        return post.w.T.dot(post.dz) * tanh_d(cur.z)


class Arctan:
    @staticmethod
    def forward(x):
        return np.arctan(x)

    @staticmethod
    def backward(cur, post):
        arctan_d = lambda x: 1 / (x * x + 1)
        return post.w.T.dot(post.dz) * arctan_d(cur.z)


class Softmax:
    @staticmethod
    def forward(x):
        return np.exp(x) / sum(np.exp(x))

    @staticmethod
    def backward(cur, post):
        return cur.a - post.a
