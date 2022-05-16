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
parser.add_argument('--save_pdf', action='store_true')

args = parser.parse_args()

with open(os.path.join(args.folder, 'results.pkl'), 'rb') as f:
    results = pickle.load(f)

epochs = []
distance = []

weights = []
weights_lin = []

loss = []
loss_lin = []
test_loss = []
test_loss_lin = []

test_output = []
test_output_lin = []

plot_random_weights = False
plot_frob_distance = False
plot_loss = False
plot_test_output = False
if 'random_weights' in results[0]:
    plot_random_weights = True
if 'frob_distance' in results[0] and 'param_norm' in results[0]:
    plot_frob_distance = True
if 'train_loss' in results[0]:
    plot_loss = True
if 'random_test_output' in results[0]:
    plot_test_output = True

for entry in results:
    if entry['epoch'] > args.max_iter:
        break
    epochs.append(entry['epoch'])

    if plot_frob_distance:
        if 'param_norm' in entry and entry['epoch'] == 0:
            init_norm = entry['param_norm']
        distance.append(entry['frob_distance'] / init_norm)

    if plot_random_weights:
        if len(weights) == 0:
            weights = [[w] for w in entry['random_weights']]
            weights_lin = [[w] for w in entry['random_weights_lin']]
        else:
            for i, w in enumerate(entry['random_weights']):
                weights[i].append(w)
            for i, w in enumerate(entry['random_weights_lin']):
                weights_lin[i].append(w)

    if plot_loss:
        loss.append(entry['train_loss'])
        loss_lin.append(entry['train_loss_lin'])
        test_loss.append(entry['test_loss'])
        test_loss_lin.append(entry['test_loss_lin'])

    if plot_test_output:
        if len(test_output) == 0:
            test_output = [[o] for o in entry['random_test_output']]
            test_output_lin = [[o] for o in entry['random_test_output_lin']]
        else:
            for i, o in enumerate(entry['random_test_output']):
                test_output[i].append(o)
            for i, o in enumerate(entry['random_test_output_lin']):
                test_output_lin[i].append(o)

# Plot normalised distance against epochs
if plot_frob_distance:
    with plt.style.context('seaborn'):
        plt.plot(epochs, distance)
        plt.xlabel(r'$t$')
        plt.ylabel(r'$\frac{\| \theta_t - \theta_0\|_F}{\|\theta_0\|_F}$')
        if args.save_pdf:
            plt.savefig(os.path.join(args.folder, f'frob_plot.pdf'))
        plt.show()

# Plot change in weights over epochs
if plot_random_weights:
    with plt.style.context('seaborn'):
        n = args.num_weights
        for w_path, w_path_lin in zip(weights[:n], weights_lin[:n]):
            lines = plt.plot(epochs, np.array(w_path) - w_path[0])
            lines_lin = plt.plot(epochs, w_path_lin, '--')
            lines_lin[0].set_color(lines[0].get_color())
        plt.xlabel(r'$t$')
        plt.ylabel('Weight Change')
        if args.save_pdf:
            plt.savefig(os.path.join(args.folder, f'weights_plot.pdf'))
        plt.show()

# Plot loss over epochs
if plot_loss:
    with plt.style.context('seaborn'):
        train_lines = plt.plot(epochs, loss, label='Train')
        test_lines = plt.plot(epochs, test_loss, label='Test')
        train_lin_lines = plt.plot(epochs, loss_lin, linestyle='dashed')
        test_lin_lines = plt.plot(epochs, test_loss_lin, linestyle='dashed')
        train_lin_lines[0].set_color(train_lines[0].get_color())
        test_lin_lines[0].set_color(test_lines[0].get_color())
        plt.xlabel(r'$t$')
        plt.ylabel('Train/Test Loss')
        plt.legend()
        if args.save_pdf:
            plt.savefig(os.path.join(args.folder, f'loss_plot.pdf'))
        plt.show()

# Plot test output over epochs
if plot_test_output:
    with plt.style.context('seaborn'):
        n = args.num_test_points
        for output_path in test_output[:n]:
            lines = plt.plot(epochs, output_path)
        plt.gca().set_prop_cycle(None)
        for output_path_lin in test_output_lin[:n]:
            plt.plot(epochs, output_path_lin, '--')
        plt.xlabel(r'$t$')
        plt.ylabel('Test Output Value')
        if args.save_pdf:
            plt.savefig(os.path.join(args.folder, f'test_output_plot.pdf'))
        plt.show()
