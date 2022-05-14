""" Compare different optimizers and plot the results."""
import os
import json
import pickle
from collections import defaultdict

import numpy as np
import torch
import argparse

from torch.utils.data import random_split
import matplotlib.pyplot as plt
from tqdm import tqdm

from optimizers.kfac import Kfac, train as kfac_train, test as kfac_test
from paths import plots_dir
from preconditioners import settings
from preconditioners.datasets import CenteredLinearGaussianDataset, CenteredQuadraticGaussianDataset
from preconditioners.optimizers import GradientDescent, PrecondGD, PrecondGD2, PrecondGD3
from preconditioners.utils import generate_true_parameter, generate_c, MLP, generate_W_star
from datetime import datetime as dt
import warnings

# Eduard comment (Treated): The way we will test it, these are going to be the sizes:
# len(test_data) < len(train_data) < no_of_parameters_of_the_model << len(extra_data)

# region CLI provided parameters
parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', help='Number of layers', default=3, type=int)
parser.add_argument('--num_runs', help='Number of runs', default=1, type=int)
parser.add_argument('--max_iter', help='Max epochs', default=float('inf'), type=int)
parser.add_argument('--test_train_ratio', help='Test train ratio', default=1, type=int)
parser.add_argument('--extra_train_ratio', help='Extra train ratio', default=5, type=int)
parser.add_argument('--test_loss_threshold', help='Test loss threshold', type=float)
parser.add_argument('--train_size', help='Train size', default=10, type=int)
parser.add_argument('--results_dir', help='Folder path', type=str)
parser.add_argument('--width', help='Hidden channels', type=int)
parser.add_argument('--damping', help='damping', type=float, default=1.0)
parser.add_argument('--tol', help='tol', type=float, default=1e-3)
parser.add_argument('--lr', help='lr', type=float, default=1e-3)
parser.add_argument('--ro', help='ro', type=float, default=.5)
parser.add_argument('--r2', help='r2', type=float, default=1)
parser.add_argument('--d', help='d', type=float, default=10)
parser.add_argument('--print_every', help='print_every', type=int, default=100)
args = parser.parse_args()
# endregion

# region CLI argument checks
assert args.num_layers >= 2, 'Number of layers must be at least 2'
assert args.num_runs >= 1, 'Number of runs must be at least 1'
assert args.max_iter >= 1, 'Max epochs must be at least 1'
assert args.test_train_ratio >= 1, 'Test train ratio must be at least 1'
assert args.extra_train_ratio >= 1, 'Extra train ratio must be at least 1'
assert args.train_size >= 1, 'Train size must be at least 1'
assert args.width >= 1, 'Hidden channels must be at least 1'
assert args.damping >= 0, 'Damping must be at least 0'
assert args.tol >= 0, 'Tolerance must be at least 0'
assert args.lr >= 0, 'Learning rate must be at least 0'
assert args.ro >= 0, 'Regularization parameter must be at least 0'
assert args.r2 >= 0, 'Regularization parameter must be at least 0'
assert args.d >= 0, 'Regularization parameter must be at least 0'
assert args.print_every >= 1, 'Print every must be at least 1'

# endregion

# region Fixed & Derived variables
# Fixed
loss_function = torch.nn.MSELoss()
noise_variances = np.linspace(1, 10, 20)
optimizer_classes = [Kfac, GradientDescent, PrecondGD]

# Derived
extra_size = args.extra_train_ratio * args.train_size
test_size = args.test_train_ratio * args.train_size
model = MLP(in_channels=args.d, num_layers=args.num_layers, hidden_channels=args.width).double().to(settings.DEVICE)
# endregion


# region Helper functions
def instantiate_optimizer(optimizer_class, train_data, extra_data):
    if optimizer_class in {PrecondGD, PrecondGD2, PrecondGD3}:
        labeled_data = train_data[:][0].double().to(settings.DEVICE)
        unlabeled_data = extra_data[:][0].double().to(settings.DEVICE)
        return optimizer_class(model, lr=args.lr, labeled_data=labeled_data, unlabeled_data=unlabeled_data,
                               damping=args.damping)
    elif optimizer_class == GradientDescent:
        return GradientDescent(model.parameters(), lr=args.lr)


def train(model, train_data, optimizer, loss_function, tol, max_iter=float('inf'), print_every=10):
    """Train the model until loss is minimized."""
    model_logs = {'condition': None, 'losses': []}
    model.train()
    current_loss = float('inf')
    epoch = 0
    # stop if 5 consecutive epochs have no improvement
    no_improvement_counter = 0
    condition = None

    while not condition:
        model.train()

        previous_loss = current_loss
        # previous_model_state = model.state_dict()

        # Get and prepare inputs
        inputs, targets = train_data[:]
        # Set the inputs and targets to the device
        inputs, targets = inputs.double().to(settings.DEVICE), targets.double().to(settings.DEVICE)
        targets = targets.reshape((targets.shape[0], 1))

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

        # Update statistics
        current_loss = loss.item()

        epoch += 1

        # Print statistics
        if epoch == 1 or epoch % print_every == 0:
            print(f'Epoch {epoch}: Train loss: {current_loss:.4f}')

        # Update condition
        delta_loss = current_loss - previous_loss
        no_improvement_counter += 1 if np.abs(delta_loss) < 1e-6 else 0
        if no_improvement_counter > 5:  # stagnation
            condition = 'stagnation'
        elif current_loss <= tol:
            condition = 'tol'
        elif epoch >= max_iter:
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


def generate_linear_params():
    w_star = generate_true_parameter(args.d, args.r2, m=np.eye(args.d))
    c = generate_c(args.ro, regime='autoregressive', d=args.d)
    return w_star, c


def generate_linear_data(w_star, c, sigma2):
    dataset = CenteredLinearGaussianDataset(w_star=w_star, d=args.d, c=c, n=args.train_size + test_size + extra_size,
                                            sigma2=sigma2)
    train_data, test_data, extra_data = random_split(dataset, [args.train_size, test_size, extra_size])

    return train_data, test_data, extra_data


def generate_quadratic_params():
    w_star = generate_true_parameter(args.d, args.r2, m=np.eye(args.d))
    W_star = generate_W_star(args.d, args.r2)
    c = generate_c(args.ro, regime='autoregressive', d=args.d)
    return W_star, w_star, c


def generate_quad_data(W_star, w_star, c, sigma2):
    dataset = CenteredQuadraticGaussianDataset(
        W_star=W_star, w_star=w_star, d=args.d, c=c,
        n=args.train_size + test_size + extra_size, sigma2=sigma2)

    train_data, test_data, extra_data = random_split(dataset, [args.train_size, test_size, extra_size])

    return train_data, test_data, extra_data


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


def plot_and_save_results(test_errors, results_dir):
    # Compute mean test errors
    mean_test_errors = defaultdict(list)
    for optim_cls, test_losses in test_errors.items():
        test_losses = np.array(test_losses)
        test_losses = test_losses.reshape(test_losses.shape[0], test_losses.shape[1])
        mean_test_errors[optim_cls] = np.nanmean(test_losses, axis=0).tolist()

    # Plot the mean test errors
    for optim_cls in optimizer_classes:
        plt.scatter(noise_variances, mean_test_errors[optim_cls.__name__], label=optim_cls.__name__)

        plt.xlabel('Noise variance')
        plt.ylabel('Test loss')
        plt.legend([optim_cls.__name__ for optim_cls in optimizer_classes])

    # Save figure
    plt.savefig(os.path.join(results_dir, 'plot.pdf'), format='pdf')

    # Save all test_errors
    with open(os.path.join(results_dir, 'test_errors.pkl'), 'wb') as f:
        pickle.dump(test_errors, f)
    # Save mean_test_errors
    with open(os.path.join(results_dir, 'mean_test_errors.pkl'), 'wb') as f:
        pickle.dump(mean_test_errors, f)

    # Print path to results
    print(f'Results saved to {results_dir}')

    plt.show()

def create_results_dir_and_save_params():
    # Create folder name
    args_dict = vars(args)
    dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    folder_name = dtstamp + '_' + '_'.join([f'{k}={v}' for k, v in args_dict.items() if v])
    results_dir = os.path.join(plots_dir, 'optimizer_benchmark', folder_name)
    
    # Create folder
    os.makedirs(results_dir)
    os.makedirs(os.path.join(results_dir, 'model_logs'))

    # Save params
    with open(os.path.join(results_dir, 'params.json'), 'w') as f:
        json.dump(args_dict, f)

    # Return results directory
    return results_dir

# endregion


if __name__ == '__main__':
    # Create results dir and save params
    results_dir = create_results_dir_and_save_params()

    # Test errors collector
    test_errors = defaultdict(list)
    
    # Generate true parameters
    W_star, w_star, c = generate_quadratic_params()
    
    # Set progress bar
    pbar = tqdm(total=args.num_runs * len(optimizer_classes) * len(noise_variances))
    pbar.set_description('Running experiments')

    # Shutdown warnings
    warnings.filterwarnings("ignore")

    # Run experiments
    for num_run in range(1, 1 + args.num_runs):
        print(f'\n\nRun N°: {num_run}')
        run_test_errors = defaultdict(list)
        for sigma2 in noise_variances:
            # Generate quadratic data
            train_data, test_data, extra_data = generate_quad_data(W_star=W_star, w_star=w_star, c=c, sigma2=sigma2)
            print(f'\n\nNoise variance: {sigma2}')
            for optim_cls in optimizer_classes:
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
                    optimizer = instantiate_optimizer(optim_cls, train_data, extra_data)
                    # Train
                    train_loss, model_logs = train(model, train_data, optimizer, loss_function, args.tol, args.max_iter,
                                                   print_every=args.print_every)
                    # Test
                    test_loss = test(model, test_data, loss_function)
                    model.reset_parameters()

                # The following line is to discard outliers w.r.t. to a given threshold
                # To deactivate, just don't set the threshold
                test_loss = float('nan') if (args.test_loss_threshold and test_loss > args.test_loss_threshold) else test_loss

                model_logs['test_loss'] = test_loss
                save_model_logs(model_logs, results_dir, model_name)

                # Add this test loss to the current run results
                run_test_errors[optim_cls.__name__].append(test_loss)

                # Update progress bar
                pbar.update(1)

        # Add this run test errors to the all test errors
        for optim_cls, test_losses in run_test_errors.items():
            test_errors[optim_cls].append(test_losses)

    # Plot and Save mean test errors
    plot_and_save_results(test_errors, results_dir)
