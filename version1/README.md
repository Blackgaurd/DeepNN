# Version 1

**Main focus**: getting a working model without using any ML libraries

The network (`Network.py`) currently has a single hidden layer, with variable parameters.

Uses ReLU for activation, but sigmoid functions are also present.

No special optimizations were used for this version. Its training is based on mini-batch gradient descent.

## Networks

The networks folder has a series of folders each with several files:

* `params.yaml`: parameters used for training the neural network
* `results.yaml`: results of the neural network based on the parameters
* `submission.csv`: submission file submitted to Kaggle