"""Plots results from the training_comparison experiment"""
import argparse
import os
import pickle
import json
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Folder of experiment', type=str)
parser.add_argument('--line_of_best_fit', action='store_true')

args = parser.parse_args()

var_vals = []
loss = []
loss_extra = []
test_loss = []
test_loss_extra = []

for experiment_folder in os.listdir(args.folder):
    with open(os.path.join(args.folder, experiment_folder, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(args.folder, experiment_folder, 'params.json'), 'rb') as f:
        params = json.load(f)
    if len(results) == 0:
        continue

    print(f"Analysing sigma2={params['sigma2']}")
    print(experiment_folder)

    var_vals.append(params['sigma2'])
    last_entry = results[-1]
    loss.append(last_entry['train_loss'])
    loss_extra.append(last_entry['train_loss_extra'])
    test_loss.append(last_entry['test_loss'])
    test_loss_extra.append(last_entry['test_loss_extra'])

# Order data by width
_, loss = zip(*sorted(zip(var_vals, loss)))
_, loss_extra = zip(*sorted(zip(var_vals, loss_extra)))
_, test_loss = zip(*sorted(zip(var_vals, test_loss)))
var_vals_plot, test_loss_extra = zip(*sorted(zip(var_vals, test_loss_extra)))

# Plot loss over width
# train_lines = plt.plot(w_plot, loss, label='Train')
# test_lines = plt.plot(w_plot, test_loss, label='Test')
# train_lin_lines = plt.plot(w_plot, loss_lin, linestyle='dashed')
# test_lin_lines = plt.plot(w_plot, test_loss_lin, linestyle='dashed')
# train_lin_lines[0].set_color(train_lines[0].get_color())
# test_lin_lines[0].set_color(test_lines[0].get_color())
# plt.xlabel(r'$n$')
# plt.ylabel('Train/Test Loss')
# plt.legend()
# plt.show()

# Plot test loss against variance
plt.scatter(var_vals_plot, test_loss, label="KFAC")
plt.scatter(var_vals_plot, test_loss_extra, label="KFAC-extra")
plt.ylabel("Test Loss")
plt.xlabel(r'$\sigma^2$')
if args.line_of_best_fit:
    plt.gca().set_prop_cycle(None)
    x = np.linspace(var_vals_plot[0], var_vals_plot[-1], 10)
    best_fit = np.poly1d(np.polyfit(var_vals_plot, test_loss, 1))(x)
    best_fit_extra = np.poly1d(np.polyfit(var_vals_plot, test_loss_extra, 1))(x)
    plt.plot(x, best_fit)
    plt.plot(x, best_fit_extra)
plt.legend()
plt.show()
