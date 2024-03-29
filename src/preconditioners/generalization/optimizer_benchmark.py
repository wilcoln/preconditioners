"""Compares the variance attained by different optimizers

Example:
python src/preconditioners/generalization/optimizer_benchmark.py --num_layers 3 --width 64
"""
import math
import os
import json
import pickle
from collections import defaultdict
from copy import deepcopy

import numpy as np
import torch
import argparse

from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from preconditioners.optimizers.kfac import Kfac, train as kfac_train, test as kfac_test
from preconditioners.paths import plots_dir
from preconditioners import settings
from preconditioners.datasets import NumpyDataset, DataGenerator
from preconditioners.optimizers import GradientDescent, PrecondGD, PrecondGD2, PrecondGD3
from preconditioners.utils import SLP, MLP
from datetime import datetime as dt
import warnings

# Fixed
LOSS_FUNCTION = torch.nn.MSELoss()
OPTIMIZER_CLASSES = [PrecondGD, GradientDescent]

def get_args():
    # CLI provided parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_runs', help='Number of runs', default=1, type=int)
    parser.add_argument('--max_iter', help='Max epochs', default=float('inf'), type=int)
    parser.add_argument('--max_variance', help='Max var', default=10, type=float)
    parser.add_argument('--min_variance', help='Min var', default=1, type=float)
    parser.add_argument('--num_plot_points', help='Number of plot points', default=10, type=int)
    # Dataset size
    parser.add_argument('--train_size', help='Train size', default=128, type=int)
    parser.add_argument('--test_size', help='Test size', default=128, type=int)
    parser.add_argument('--extra_size', help='Extra size', default=128, type=int)
    # Model/opt parameters
    parser.add_argument('--num_layers', help='Number of layers', default=3, type=int)
    parser.add_argument('--width', help='Hidden channels', type=int, default=8)
    parser.add_argument('--damping', help='damping', type=float, default=1.0)
    parser.add_argument('--tol', help='tol', type=float, default=1e-3)
    parser.add_argument('--lr', help='lr', type=float, default=1e-3)
    parser.add_argument('--gd_lr', help='gd lr', type=float, default=None)
    parser.add_argument('--stagnation_threshold', help='Maximum change in loss that counts as no progress', type=float, default=1e-6)
    parser.add_argument('--stagnation_count_max', help='Maximum number of iterations of no progress before the experiment terminates', type=int, default=5)
    # Data parameters
    parser.add_argument('--dataset', help='Type of dataset', choices=['linear', 'quadratic', 'MLP'], default='quadratic')
    parser.add_argument('--ro', help='ro', type=float, default=.5)
    parser.add_argument('--r2', help='r2', type=float, default=1)
    parser.add_argument('--d', help='d', type=float, default=10)
    parser.add_argument('--use_init_fisher', action='store_true')
    parser.add_argument('--fisher_update_steps', type=int, default=None)
    parser.add_argument('--lr_multiplier', type=float, default=1)
    parser.add_argument('--print_every', help='print_every', type=int, default=100)
    args = parser.parse_args()
    # endregion

    # region CLI argument checks
    assert args.num_layers >= 2, 'Number of layers must be at least 2'
    assert args.num_runs >= 1, 'Number of runs must be at least 1'
    assert args.max_iter >= 1, 'Max epochs must be at least 1'
    assert args.test_size >= 1, 'Test train ratio must be at least 1'
    assert args.extra_size >= 1, 'Extra train ratio must be at least 1'
    assert args.train_size >= 1, 'Train size must be at least 1'
    assert args.width >= 1, 'Hidden channels must be at least 1'
    assert args.damping >= 0, 'Damping must be at least 0'
    assert args.tol >= 0, 'Tolerance must be at least 0'
    assert args.lr >= 0, 'Learning rate must be at least 0'
    assert args.ro >= 0, 'Regularization parameter must be at least 0'
    assert args.r2 >= 0, 'Regularization parameter must be at least 0'
    assert args.d >= 0, 'Regularization parameter must be at least 0'
    assert args.print_every >= 1, 'Print every must be at least 1'

    return args

# region Helper functions
def instantiate_optimizer(optimizer_class, train_data, extra_data, lr=1, gd_lr=None, damping=1, use_init_fisher=False):
    if optimizer_class in {PrecondGD, PrecondGD2, PrecondGD3}:
        labeled_data = train_data[:][0].double().to(settings.DEVICE)
        unlabeled_data = extra_data[:][0].double().to(settings.DEVICE)
        return optimizer_class(model, lr=lr, labeled_data=labeled_data, unlabeled_data=unlabeled_data,
                               damping=damping, is_linear=use_init_fisher)
    elif optimizer_class == GradientDescent:
        lr = gd_lr if gd_lr is not None else lr
        return GradientDescent(model.parameters(), lr=lr)

def train(model, train_data, optimizer, loss_function, args):
    """Train the model until loss is minimized."""
    model_logs = {'condition': None, 'losses': []}
    model.train()
    current_loss = float('inf')
    epoch = 0
    # stop if 5 consecutive epochs have no improvement
    no_improvement_counter = 0
    condition = None
    stag_loss = float('inf')

    # Get and prepare inputs
    inputs, targets = train_data[:]
    # Set the inputs and targets to the device
    inputs, targets = inputs.double().to(settings.DEVICE), targets.double().to(settings.DEVICE)
    targets = targets.reshape((targets.shape[0], 1))

    while not condition:
        model.train()

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = model(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        epoch += 1

        if args.fisher_update_steps is not None and args.use_init_fisher and epoch % args.fisher_update_steps == 0 and isinstance(optimizer, PrecondGD):
            optimizer.last_p_inv = None
            # change the learning rate
            for group in optimizer.param_groups:
                group['lr'] = group['lr'] * args.lr_multiplier

        # Print statistics
        if epoch == 1 or epoch % args.print_every == 0:
            # Update statistics
            current_loss = loss.item()

            print(f'Epoch {epoch}: Train loss: {current_loss:.4f}')

            # Update condition
            delta_loss = stag_loss - current_loss
            if delta_loss < args.stagnation_threshold:
                no_improvement_counter += 1
                stag_loss = min(stag_loss, current_loss)
            else:
                no_improvement_counter = 0
                stag_loss = current_loss


            if no_improvement_counter > args.stagnation_count_max:  # stagnation
                condition = 'stagnation'
            elif current_loss <= args.tol:
                condition = 'tol'
            elif epoch >= args.max_iter:
                condition = 'max_iter'

            model_logs['losses'].append(current_loss)

    # Final print
    print('*** FINAL EPOCH ***')
    print(f'Epoch {epoch}: Train loss: {current_loss:.4f}, Stop condition: {condition}')
    #
    # Save train logs
    model_logs['condition'] = condition

    # Return loss
    return current_loss, model_logs


def test(model, test_data, loss_function):
    """Test the model."""
    model.eval()
    with torch.no_grad():
        inputs, targets = test_data[:]
        inputs, targets = inputs.double().to(settings.DEVICE), targets.double().to(settings.DEVICE)
        targets = targets.reshape((targets.shape[0], 1))
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
    return loss.item()


def save_model_logs(model_logs, results_dir, model_name):
    # Plot train loses
    plt.title(model_name + ' | ' + model_logs['condition'])
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.plot(model_logs['losses'])

    # Save train losses plot
    plt.savefig(os.path.join(results_dir, 'model_logs', f'{model_name}_train.pdf'))
    plt.close()

    # Save model logs dict(losses=List[float], condition=str, test_loss=float)
    with open(os.path.join(results_dir, 'model_logs', f'{model_name}_logs.pkl'), 'wb') as f:
        pickle.dump(model_logs, f)


def plot_and_save_results(test_errors, results_dir, save_test_error=True, final_plot=False):
    # Compute mean test errors
    mean_test_errors = defaultdict(list)
    for optim_cls, test_losses in test_errors.items():
        test_losses = np.array(test_losses)
        test_losses = test_losses.reshape(test_losses.shape[0], test_losses.shape[1])
        mean_test_errors[optim_cls] = np.nanmean(test_losses, axis=0).tolist()

    # Plot the mean test errors
    for optim_cls in OPTIMIZER_CLASSES:
        plt.scatter(noise_variances, mean_test_errors[optim_cls.__name__], label=optim_cls.__name__)

        plt.xlabel('Noise variance')
        plt.ylabel('Test loss')
        plt.legend([optim_cls.__name__ for optim_cls in OPTIMIZER_CLASSES])

    # Save figure
    dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    if final_plot:
        plot_location = os.path.join(results_dir, 'plot_' + dtstamp + '.pdf')
    else:
        plot_location = os.path.join(results_dir, 'plots', 'plot_' + dtstamp + '.pdf')
    plt.savefig(plot_location, format='pdf')

    # Save all test_errors
    if save_test_error:
        with open(os.path.join(results_dir, 'test_errors.pkl'), 'wb') as f:
            pickle.dump(test_errors, f)
    # Save mean_test_errors
    with open(os.path.join(results_dir, 'mean_test_errors.pkl'), 'wb') as f:
        pickle.dump(mean_test_errors, f)

    # Print path to results
    print(f'\nResults saved to {results_dir}')

    if final_plot:
        plt.show()
    else:
        plt.clf()

def create_results_dir_and_save_params(params_dict):
    # Create folder name
    dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    folder_name = dtstamp
    results_dir = os.path.join(plots_dir, 'optimizer_benchmark_' + dtstamp)

    # Create folder
    os.makedirs(results_dir)
    os.makedirs(os.path.join(results_dir, 'model_logs'))
    os.makedirs(os.path.join(results_dir, 'plots'))

    # Save params
    with open(os.path.join(results_dir, 'params.json'), 'w') as f:
        json.dump(params_dict, f)

    # Return results directory
    return results_dir

# endregion


if __name__ == '__main__':
    # Set seed and log settings
    np.random.seed(404)
    torch.manual_seed(404)
    warnings.filterwarnings("ignore")

    # Collect arguments
    args = get_args()
    noise_variances = np.linspace(args.min_variance, args.max_variance, args.num_plot_points)
    num_params = (1 + args.d) * args.width + (args.width ** 2) * (args.num_layers - 2)

    # Create data generator
    data_generator = DataGenerator(args.dataset, d=args.d, regime='autoregressive',
        ro=args.ro, r1=args.r2, num_layers=args.num_layers, hidden_channels=args.width)
    # Generate extra data
    extra_data = data_generator.generate(args.extra_size, sigma2=0)
    x, y = extra_data
    extra_data = NumpyDataset(x, y)
    inv_fisher_cache = None

    # Create model
    model = MLP(in_channels=args.d, num_layers=args.num_layers, hidden_channels=args.width).double().to(settings.DEVICE)
    init_model_state = deepcopy(model.state_dict())

    # Set progress bar
    pbar = tqdm(total=args.num_runs * len(OPTIMIZER_CLASSES) * len(noise_variances))
    pbar.set_description('Running experiments')
    # Create test errors
    test_errors = defaultdict(list)
    # Create results dir and save params
    results_dir = create_results_dir_and_save_params(params_dict=vars(args))

    # Run experiments
    for num_run in range(1, 1 + args.num_runs):
        # Generate true parameters
        noiseless_data = data_generator.generate(args.train_size + args.test_size, sigma2=0)
        x, y_noiseless = noiseless_data

        # Compute signal
        average_response = np.sum(np.square(y_noiseless)) / (args.train_size + args.test_size)
        print(f"Average norm of response {average_response}")
        print(f"r^2:{args.r2}")

        run_test_errors = defaultdict(list)
        for sigma2 in noise_variances:
            print(f'\n\nRun N°: {num_run}')
            print(f'Noise variance: {sigma2}')

            # Add noise to data to create train and test sets
            xi = np.random.normal(0, np.sqrt(sigma2), size=(args.train_size + args.test_size))
            y = y_noiseless + xi
            train_data = NumpyDataset(x[:args.train_size], y[:args.train_size])
            test_data = NumpyDataset(x[args.train_size:], y[args.train_size:])

            for optim_cls in OPTIMIZER_CLASSES:
                # For each optimizer
                model_name = optim_cls.__name__ + f'_sigma2={sigma2}' + f'_run={num_run}'
                print(f'\n\nOptimizer: {optim_cls.__name__}')
                if optim_cls.__name__ == 'Kfac':  # KFAC calls jax model and optimizer
                    # Train
                    train_loss, hk_model, params, model_logs = kfac_train(
                        train_data,
                        mlp_output_sizes=[args.width] * (args.num_layers - 1) + [1],
                        max_iter=args.max_iter, damping=args.damping,
                        tol=args.tol, print_every=args.print_every
                    )
                    # Test
                    test_loss = kfac_test(hk_model, params, test_data)
                else:
                    # Instantiate the optimizer
                    optimizer = instantiate_optimizer(optim_cls, train_data,
                        extra_data, lr=args.lr, gd_lr=args.gd_lr,
                        damping=args.damping, use_init_fisher=args.use_init_fisher)

                    # For caching inverse fisher matrix
                    if optim_cls == PrecondGD and args.use_init_fisher:
                        if inv_fisher_cache is not None:
                            optimizer.last_p_inv = inv_fisher_cache
                        else:
                            inv_fisher_cache = optimizer._compute_p_inv()

                    # Train
                    train_loss, model_logs = train(model, train_data, optimizer, LOSS_FUNCTION, args)

                    # Test
                    test_loss = test(model, test_data, LOSS_FUNCTION)
                    model.reset_parameters(init_model_state)


                model_logs['sigma2'] = sigma2
                model_logs['average_response'] = average_response
                model_logs['test_loss'] = test_loss
                model_logs['optimizer'] = optim_cls.__name__
                save_model_logs(model_logs, results_dir, model_name)
                print(f"Test loss: {test_loss:4f}")

                # Add this test loss to the current run results
                run_test_errors[optim_cls.__name__].append(test_loss)

                # Update progress bar
                pbar.update(1)

        # Add this run test errors to the all test errors
        for optim_cls, test_losses in run_test_errors.items():
            test_errors[optim_cls].append(test_losses)

        # Plot and Save mean test errors
        plot_and_save_results(test_errors, results_dir, save_test_error=False, final_plot=False)

    plot_and_save_results(test_errors, results_dir, save_test_error=True, final_plot=True)
