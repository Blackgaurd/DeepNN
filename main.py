import yaml

from Network import Network

network_number = "01"

with open(f"networks/{network_number}/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

neural_net = Network("input/train.csv", **params)
#neural_net.gradient_descent(report_accuracy=50)
neural_net.make_prediction_rand()
neural_net.make_prediction_test("input/test.csv", f"network/{network_number}/submission.csv")