"""Plots results from the training_comparison experiment"""
import argparse
import os
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({"text.usetex": True})

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Folder of experiment', type=str)

args = parser.parse_args()

test_errors = {}
var_vals = []

model_logs_folder = os.path.join(args.folder, 'model_logs')

for experiment_file in os.listdir(model_logs_folder):
    if experiment_file[-4:] != ".pkl":
        continue
    with open(os.path.join(model_logs_folder, experiment_file), 'rb') as f:
        results = pickle.load(f)

    s2 = results['sigma2']
    optimizer = results['optimizer']
    test_loss = results['test_loss']

    print(f"Analysing sigma2={s2}")
    print(experiment_file)

    if optimizer not in test_errors:
        test_errors[optimizer] = defaultdict(list)

    test_errors[optimizer][s2].append(test_loss)
    var_vals.append(s2)

var_vals = sorted(set(var_vals))
# Plot test loss against variance
print(test_errors)
for optimizer in test_errors:
    average_test_error = [np.average(test_errors[optimizer][s2]) for s2 in var_vals]
    plt.scatter(var_vals, average_test_error, label=optimizer)

plt.ylabel("Test Loss")
plt.xlabel(r'$\sigma^2$')
plt.legend()
plt.show()
