import yaml
import time
import numpy as np

from network import Network

network_number = "05"

# read params
with open(f"networks/{network_number}/params.yaml", "r") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

print("Running neural network with:")
for key, val in params.items():
    print(f"{key}: {val}")
print()

# run neural network
neural_net = Network("../input/train.csv", **params)

start = time.time()
neural_net.gradient_descent(report_accuracy=50)
end = time.time()

# predict + write to submissions.csv
neural_net.predict_sub("../input/test.csv", f"networks/{network_number}/submission.csv")

# record results
with open(f"networks/{network_number}/results.yaml", "w") as f:
    time_taken = end - start
    results = {
        "time taken": round(time_taken, 4),
        "accuracy": round(float(input("How'd it do?\n>>> ")) * 100, 4),
    }

    yaml.dump(results, f, default_flow_style=False)