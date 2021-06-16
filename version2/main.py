from matplotlib import pyplot as plt
import time

from network import Network
from layer import Dense, Utils
from activation import *

Utils.seed(1)

relu = Network(
    500,
    0.1,
    [
        Dense(256, Relu),
        Dense(256, Relu),
        Dense(10, Softmax),
    ],
    train_file_path="../input/train.csv",
)

for i in relu.run(250):
    print(f"Epoch:\t\t{i[0]}/250\nAccuracy:\t{i[1]}\n")
    time.sleep(0.25)

relu.submit_csv("../input/test.csv", "submission.csv")
