"""Plots results from the training_comparison experiment"""
import argparse
import os
import pickle
import numpy as np

import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex": True})

parser = argparse.ArgumentParser()
parser.add_argument('--folder', help='Folder of experiment', type=str)
parser.add_argument('--num_test_points', help='Number of test points to plot', type=int, default=5)
parser.add_argument('--num_weights', help='Number of weights to plot', type=int, default=5)
parser.add_argument('--max_iter', help='Maximum number of iterations to plot', type=int, default=float('inf'))

args = parser.parse_args()

with open(os.path.join(args.folder, 'results.pkl'), 'rb') as f:
    results = pickle.load(f)

epochs = []
distance = []

loss = []
loss_extra = []
test_loss = []
test_loss_extra = []

plot_loss = False
if 'train_loss' in results[0]:
    plot_loss = True

for entry in results:
    if entry['epoch'] > args.max_iter:
        break
    epochs.append(entry['epoch'])

    if plot_loss:
        loss.append(entry['train_loss'])
        loss_extra.append(entry['train_loss_extra'])
        test_loss.append(entry['test_loss'])
        test_loss_extra.append(entry['test_loss_extra'])

# Plot loss over epochs
if plot_loss:
    train_lines = plt.plot(epochs, loss, label='Train')
    test_lines = plt.plot(epochs, test_loss, label='Test')
    train_extra_lines = plt.plot(epochs, loss_extra, linestyle='dashed')
    test_extra_lines = plt.plot(epochs, test_loss_extra, linestyle='dashed')
    train_extra_lines[0].set_color(train_lines[0].get_color())
    test_extra_lines[0].set_color(test_lines[0].get_color())
    plt.xlabel(r'$t$')
    plt.ylabel('Train/Test Loss')
    plt.legend()
    plt.show()
