import argparse
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Folder of experiment', type=str)

args = parser.parse_args()

with open(os.path.join(args.folder, 'results.pkl'), 'rb') as f:
    results = pickle.load(f)

epochs = []
distance = []
weights = []
weights_lin = []
loss = []
loss_lin = []
init_norm = None

for entry in results:
    if entry['epoch'] == 0 and 'param_norm' in entry:
        init_norm = entry['param_norm']
    epochs.append(entry['epoch'])
    distance.append(entry['frob_distance'])

    if len(weights) == 0 and 'random_weights' in entry:
        weights = [[w] for w in entry['random_weights']]
        weights_lin = [[w] for w in entry['random_weights_lin']]
    elif 'random_weights' in entry:
        for i, w in enumerate(entry['random_weights']):
            weights[i].append(w)
        for i, w in enumerate(entry['random_weights_lin']):
            weights_lin[i].append(w)

    loss.append(entry['train_loss'])
    loss_lin.append(entry['train_loss_lin'])

if init_norm is None:
    raise Error('No initial parameter norm found')

# Plot normalised distance against epochs
plt.plot(epochs, np.array(distance) / init_norm)
plt.xlabel(r'$t$')
plt.ylabel(r'$\frac{\| \theta_t - \theta_0\|_F}{\|\theta_0\|_F}$')
plt.show()

# Plot change in weights over epochs
for w_path, w_path_lin in zip(weights, weights_lin):
    lines = plt.plot(epochs, np.array(w_path) - w_path[0])
    lines_lin = plt.plot(epochs, w_path_lin, '--')
    lines_lin[0].set_color(lines[0].get_color())
plt.xlabel(r'$t$')
plt.ylabel('Weight change')
plt.show()

# Plot loss over epochs
plt.plot(epochs, loss, label='MLP')
plt.plot(epochs, loss_lin, label='Linearized')
plt.xlabel(r'$t$')
plt.ylabel('Train loss')
plt.show()
