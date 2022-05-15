"""Runs the kfac_training_comparison experiment for multiple values of the variance"""
import argparse
import numpy as np
from datetime import datetime as dt
import os
import pickle
import json

import jax
from jax.example_libraries import stax
from jax.nn.initializers import normal

from preconditioners.datasets import generate_data
from preconditioners.generalization.kfac_extra_training import ExtraDataExperiment

def get_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--num_layers', help='Number of layers', type=int, default=3)
    parser.add_argument('--width', help='', type=int, default=128)
    # Optimizer params
    parser.add_argument('--lr', help='Learning rate', type=float, default=1e-1)
    parser.add_argument('--damping', help='Damping coefficient', type=float, default=1e-1)
    parser.add_argument('--l2', help='L2-regularization coefficient', type=float, default=0.)
    parser.add_argument('--use_adaptive_lr', action='store_true', default=False)
    # Data params
    parser.add_argument('--dataset', help='Type of dataset', choices=['linear', 'quadratic'], default='quadratic')
    parser.add_argument('--train_size', help='Number of train examples', type=int, default=128)
    parser.add_argument('--test_size', help='Number of test examples', type=int, default=128)
    parser.add_argument('--extra_size', help='Number of test examples', type=int, default=256)
    parser.add_argument('--in_dim', help='Dimension of features', type=int, default=8)
    parser.add_argument('--sigma2_min', help='Minimum standard deviation of label noise', type=float, default=0.5)
    parser.add_argument('--sigma2_max', help='Minimum standard deviation of label noise', type=float, default=10)
    parser.add_argument('--sigma2_step', help='Minimum standard deviation of label noise', type=float, default=.5)
    # Experiment params
    parser.add_argument('--max_iter', help='Max epochs', type=int, default=256)
    parser.add_argument('--num_runs', help='Number of runs per variance value', type=int, default=1)
    parser.add_argument('--save_every', help='Number of epochs per log', type=int, default=8)
    parser.add_argument('--save_folder', help='Experiments are saved here', type=str, default="experiments/")
    parser.add_argument('--num_test_points', help='Number of test points to analyse', type=int, default=32)

    return parser.parse_args()

def create_jax_model(width, num_layers, in_dim, out_dim, key):
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
    args = get_args()
    train_size, test_size, extra_size = args.train_size, args.test_size, args.extra_size
    dataset, in_dim = args.dataset, args.in_dim
    width, num_layers = args.width, args.num_layers
    lr, damping, l2, use_adaptive_lr = args.lr, args.damping, args.l2, args.use_adaptive_lr
    max_iter, save_every, num_test_points = args.max_iter, args.save_every, args.num_test_points
    ro, r1, regime = 0.5, 1, 'autoregressive'
    min_var, max_var, step_var = args.sigma2_min, args.sigma2_max, args.sigma2_step
    save_folder, num_runs = args.save_folder, args.num_runs

    # Generate data
    print("Generating data...")

    params_dict = vars(args)
    params_dict['regime'] = regime
    params_dict['ro'] = ro
    params_dict['r1'] = r1

    dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    experiment_name = "kfac_extra_variance_" + dtstamp
    experiment_folder = os.path.join(save_folder, experiment_name)
    os.makedirs(experiment_folder)

    # Generate data
    noiseless_data = generate_data(dataset, n=train_size + extra_size + test_size,
        d=in_dim, regime='autoregressive', ro=ro, r1=r1, sigma2=0)
    x, y = noiseless_data
    y = np.expand_dims(y, 1)
    extra_data = (x[-extra_size:], y[-extra_size:])
    x = x[:-extra_size]
    y_noiseless = y[:-extra_size]

    # Create MLP
    key = jax.random.PRNGKey(42)
    model, init_params = create_jax_model(width, num_layers, in_dim=in_dim, out_dim=1, key=key)

    # For each variance value, run the KFAC training comparison experiment
    for variance in np.arange(min_var, max_var + step_var, step_var):
        variance = variance.astype(float)
        for i in range(num_runs):
            # Add noise to labels
            xi = np.random.normal(0, np.sqrt(variance), size=(train_size + test_size, 1))
            y = y_noiseless + xi
            train_data = (x[:train_size], y[:train_size])
            test_data = (x[train_size:], y[train_size:])

            print(f"\nPerforming expeirment for sigma2={variance}...")
            params_dict['sigma2'] = variance

            # Set up experiment
            print("Setting up optimizers...")
            params = init_params
            experiment = ExtraDataExperiment(model, params, lr=lr, damping=damping, l2=l2, use_adaptive_lr=use_adaptive_lr)
            key, sub_key = jax.random.split(key)
            experiment.setup(train_data, extra_data, sub_key)

            # Run experiment
            print("Starting training...")
            experiment.log_result(train_data, test_data)
            while experiment.epoch < max_iter:
                key, sub_key = jax.random.split(key)
                experiment.step(train_data, extra_data, sub_key)

                if experiment.epoch % save_every == 0:
                    experiment.log_result(train_data, test_data)

            # Save results
            experiment.save_results(experiment_folder, params_dict)
            print("Saved results")
    print("Finished")
