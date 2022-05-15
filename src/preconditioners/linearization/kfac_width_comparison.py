"""Runs the kfac_training_comparison experiment for multiple n"""
import argparse
import numpy as np
from datetime import datetime as dt
import os
import pickle
import json

import jax
from jax.example_libraries import stax
from jax.nn.initializers import normal

from preconditioners.datasets import generate_data, data_random_split
from preconditioners.linearization.kfac_training_comparison import LinearizationExperiment

def create_argparser():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--num_layers', help='Number of layers', type=int, default=3)
    parser.add_argument('--min_width', help='Minimum number of hidden channels', type=int, default=64)
    parser.add_argument('--max_width', help='Maximum number of hidden channels', type=int, default=1024)
    parser.add_argument('--step_width', help='Step between widths', type=int, default=64)
    # Optimizer params
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-4)
    parser.add_argument('--damping', help='Damping coefficient', type=float, default=1e-3)
    parser.add_argument('--l2', help='L2-regularization coefficient', type=float, default=0.)
    # Data params
    parser.add_argument('--dataset', help='Type of dataset', choices=['linear', 'quadratic'], default='quadratic')
    parser.add_argument('--train_size', help='Number of train examples', type=int, default=128)
    parser.add_argument('--test_size', help='Number of test examples', type=int, default=128)
    parser.add_argument('--in_dim', help='Dimension of features', type=int, default=8)
    parser.add_argument('--sigma2', help='Standard deviation of label noise', type=float, default=1)
    # Experiment params
    parser.add_argument('--max_iter', help='Max epochs', type=int, default=256)
    parser.add_argument('--save_every', help='Number of epochs per log', type=int, default=8)
    parser.add_argument('--save_folder', help='Experiments are saved here', type=str, default="experiments/")
    parser.add_argument('--num_test_points', help='Number of test points to analyse', type=int, default=32)

    return parser.parse_args()

def create_model(width, num_layers, in_dim, out_dim, key):
    """Creates MLP and square loss function using stax"""
    layers = []
    scaling = 1./np.sqrt(width)
    for i in range(num_layers - 1):
        layers.append(stax.Dense(width, W_init=normal(scaling)))
        layers.append(stax.Tanh)
    layers.append(stax.Dense(out_dim, W_init=normal(scaling)))

    init_fn, f = stax.serial(*layers)

    _, params = init_fn(key, (-1, in_dim))

    return f, params

if __name__ == "__main__":
    args = create_argparser()
    train_size, test_size = args.train_size, args.test_size
    dataset, in_dim = args.dataset, args.in_dim
    min_width, max_width, step_width, num_layers = args.min_width, args.max_width, args.step_width, args.num_layers
    lr, damping, l2 = args.lr, args.damping, args.l2
    max_iter, save_every, num_test_points = args.max_iter, args.save_every, args.num_test_points
    ro, r1, sigma2, regime = 0.5, 1, args.sigma2, 'autoregressive'
    save_folder = args.save_folder

    # Generate data
    print("Generating data...")
    dataset = generate_data(dataset, n=train_size + test_size, d=in_dim,
        regime='autoregressive', ro=0.5, r1=1, sigma2=1)
    train_data, test_data = data_random_split(dataset, (train_size, test_size))

    # Choose random test points to analyse
    x_test, y_test = test_data
    test_indices = np.random.randint(0, x_test.shape[0], size=num_test_points)
    x_rand_test = np.array([x_test[i] for i in test_indices])

    params_dict = vars(args)
    params_dict['regime'] = regime
    params_dict['ro'] = ro
    params_dict['r1'] = r1

    dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    experiment_name = "width_comparison_" + dtstamp
    experiment_folder = os.path.join(save_folder, experiment_name)
    os.makedirs(experiment_folder)

    # For each width value, run the KFAC training comparison experiment
    for width in range(min_width, max_width + 1, step_width):
        print(f"\nPerforming expeirment for n={width}...")

        width_lr = lr
        width_damping = damping * width

        params_dict['width'] = width
        params_dict['lr'] = width_lr
        params_dict['damping'] = width_damping

        # Create MLP
        key = jax.random.PRNGKey(42)
        model, params = create_model(width, num_layers, in_dim=in_dim, out_dim=1, key=key)

        # Set up experiment
        print("Setting up optimizers...")
        experiment = LinearizationExperiment(model, params, lr=width_lr, damping=width_damping, l2=l2)
        key, sub_key = jax.random.split(key)
        experiment.setup(train_data, sub_key)

        # Run experiment
        print("Starting training...")
        experiment.log_result(train_data, test_data, x_rand_test)
        while experiment.epoch < max_iter:
            key, sub_key = jax.random.split(key)
            experiment.step(train_data, sub_key)

            if experiment.epoch % save_every == 0:
                experiment.log_result(train_data, test_data, x_rand_test)

        # Save results
        experiment.save_results(experiment_folder, params_dict)
        print("Saved results")
    print("Finished")
