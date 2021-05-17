# Digit Recognizer

Digit recognizer neural network made for [this](kaggle.com/c/digit-recognizer). Uses MNIST data.

The network (`Network.py`) currently has a single hidden layer, with parameters for variable parameters.

Uses ReLU for activation, but sigmoid functions are also present.

## `__init__`

Constructor for neural network.

Parameters:

* `train_file_path: str`: file path for the training images/numbers
* `hidden_layer_size: int`: number of nodes for the hidden layer
* `batch_size: int`: size of mini-batches used for training
* `iterations: int`: iterations in gradient descent
* `learning_rate: float`: learning rate (speed) of the network; more speed at the comprise of accuracy and vice versa

## `gradient_descent`

What actually does the calculations and training. A wrapper function for forward and backward propagation.

Parameters:

* `report_accuracy: int`: print an update for network accuracy every n iterations

## `predict_rand`

Makes for cool demos.

Parameters:

* `show_image: bool`: wether or not to display image and predictions

## `predict_sub`

Make predictions with test data. Write to a .csv file following format specified by Kaggle.

Parameters:

* `test_file_path: str`: file path for test images/numbers
* `output_file_path: str`: file path for where to output results

---

## Experiments

Under the 'networks' folder you can find different sets of parameters used for training, and how well they performed on the test data.

Note that results may vary depending on hardware and the random seed.

Each folder contains:

* `params.yaml`: parameters used for training
* `results.yaml`:
  * `time taken`: time taken for gradient descent, in seconds
  * `accuracy`: results of submissions according to Kaggle
* `submission.csv`: submission submitted to Kaggle
