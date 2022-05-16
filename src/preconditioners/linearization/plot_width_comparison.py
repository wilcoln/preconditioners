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
parser.add_argument('--stagnation_threshold', help='Maximum change in loss that counts as no progress', type=float, default=0.)
parser.add_argument('--stagnation_count_max', help='Maximum number of iterations of no progress before the experiment terminates', type=int, default=5)
parser.add_argument('--max_iter', help='Maximum number of iterations to plot', type=int, default=float('inf'))
parser.add_argument('--min_width', help='Minimum width to plot', type=int, default=0)
parser.add_argument('--max_width', help='Maximum width to plot', type=int, default=float('inf'))
parser.add_argument('--save_pdf', action='store_true')

args = parser.parse_args()

widths = []
loss = []
test_loss = []
loss_lin = []
test_loss_lin = []

test_output = []
test_output_lin = []

test_output_dist = []

stag_thresh, stag_count_max = args.stagnation_threshold, args.stagnation_count_max

for experiment_folder in os.listdir(args.folder):
    if not os.path.isdir(os.path.join(args.folder, experiment_folder)):
        continue
    with open(os.path.join(args.folder, experiment_folder, 'results.pkl'), 'rb') as f:
        results = pickle.load(f)
    with open(os.path.join(args.folder, experiment_folder, 'params.json'), 'rb') as f:
        params = json.load(f)
    if len(results) == 0:
        continue

    if params['width'] > args.max_width or params['width'] < args.min_width:
        continue

    print(f"Analysing n={params['width']}")
    print(experiment_folder)

    widths.append(params['width'])

    prev_entry = results[0]
    stagnation_count = 0
    for entry in results:
        if entry['epoch'] >= args.max_iter:
            print("Max iter reached: terminating early")
            break

        if stag_thresh > 0:
            # Check if train loss is stagnating
            delta_epoch = entry['epoch'] - prev_entry['epoch']
            if abs(prev_entry['train_loss'] - entry['train_loss']) < stag_thresh * delta_epoch:
                stagnation_count += delta_epoch
                if stagnation_count >= stag_count_max:
                    print(f"Stagnation: terminating early at epoch {entry['epoch']}")
                    break
            else:
                stagnation_count = 0

        prev_entry = entry

    terminal_entry = entry

    loss.append(terminal_entry['train_loss'])
    loss_lin.append(terminal_entry['train_loss_lin'])
    test_loss.append(terminal_entry['test_loss'])
    test_loss_lin.append(terminal_entry['test_loss_lin'])

    out = terminal_entry['random_test_output']
    out_lin = terminal_entry['random_test_output_lin']
    test_output.append(out)
    test_output_lin.append(out_lin)

    dist = sum([(t - t_lin)**2 for t, t_lin in zip(out, out_lin)])**.5
    test_output_dist.append(dist)

# Order data by width
_, loss = zip(*sorted(zip(widths, loss)))
_, loss_lin = zip(*sorted(zip(widths, loss_lin)))
_, test_loss = zip(*sorted(zip(widths, test_loss)))
w_plot, test_loss_lin = zip(*sorted(zip(widths, test_loss_lin)))

# Plot loss over width
with plt.style.context('seaborn'):
    train_lines = plt.plot(w_plot, loss, label='Train')
    test_lines = plt.plot(w_plot, test_loss, label='Test')
    train_lin_lines = plt.plot(w_plot, loss_lin, linestyle='dashed')
    test_lin_lines = plt.plot(w_plot, test_loss_lin, linestyle='dashed')
    train_lin_lines[0].set_color(train_lines[0].get_color())
    test_lin_lines[0].set_color(test_lines[0].get_color())
    plt.xlabel(r'$n$')
    plt.ylabel('Train/Test Loss')
    plt.legend()
    if args.save_pdf:
        plt.savefig(os.path.join(args.folder, f'train_test_plot.pdf'))
    plt.show()

# Order data by width
_, test_output = zip(*sorted(zip(widths, test_output)))
_, test_output_lin = zip(*sorted(zip(widths, test_output_lin)))

# Order data by width
_, test_output_dist = zip(*sorted(zip(widths, test_output_dist)))
with plt.style.context('seaborn'):
    plt.plot(w_plot, test_output_dist)
    plt.ylabel(r'$\|f(x) - f_{lin}(x)\|_2$')
    plt.xlabel(r'$n$')
    if args.save_pdf:
        plt.savefig(os.path.join(args.folder, f'test_output_plot.pdf'))
    plt.show()
