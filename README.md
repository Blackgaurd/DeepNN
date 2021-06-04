# Digit Recognizer

Digit recognizer neural network made for [this](kaggle.com/c/digit-recognizer).

As the name suggests, it can recognize hand written digits.

## How to use

First, install required libraries:

```$ pip install requirements.txt```

Under each version folder there is a `main.py`. At the top of the file there will be a variable for `network_number`.

Create a new folder under the respective `networks` folder, and copy the folder name into `network_number`.

Under the current network folder, create a file called `params.yaml`. A template can be found under `version_number\networks\00\params.yaml`.

Run `main.py`.

```python main.py```