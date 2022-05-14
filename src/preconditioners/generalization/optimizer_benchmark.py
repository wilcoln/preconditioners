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

from optimizers.kfac import Kfac, train as kfac_train, test as kfac_test
from paths import plots_dir
from preconditioners import settings
from preconditioners.datasets import CenteredLinearGaussianDataset, CenteredQuadraticGaussianDataset
from preconditioners.optimizers import GradientDescent, PrecondGD, PrecondGD2, PrecondGD3
from preconditioners.utils import generate_true_parameter, generate_c, MLP, generate_W_star
from datetime import datetime as dt


# Eduard comment (Treated): The way we will test it, these are going to be the sizes:
# len(test_data) < len(train_data) < no_of_parameters_of_the_model << len(extra_data)

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', help='Number of layers', default=1, type=int)
parser.add_argument('--num_runs', help='Number of runs', default=1, type=int)
parser.add_argument('--max_iter', help='Max epochs', default=float('inf'), type=int)
parser.add_argument('--test_train_ratio', help='Test train ratio', default=1, type=int)
parser.add_argument('--test_loss_threshold', help='Test loss threshold', type=float)
parser.add_argument('--train_size', help='Train size', default=10, type=int)
parser.add_argument('--folder_path', help='Folder path', type=str)
parser.add_argument('--width', help='Hidden channels', type=int)
parser.add_argument('--damping', help='damping', type=float, default=1.0)
parser.add_argument('--tol', help='tol', type=float, default=1e-3)
parser.add_argument('--print_every', help='print_every', type=int, default=100)
args = parser.parse_args()


# Helper functions
def instantiate_optimizer(optimizer_class, train_data, extra_data):
    if optimizer_class in {PrecondGD, PrecondGD2, PrecondGD3}:
        labeled_data = train_data[:][0].double().to(settings.DEVICE)
        unlabeled_data = extra_data[:][0].double().to(settings.DEVICE)
        return optimizer_class(model, lr=lr, labeled_data=labeled_data, unlabeled_data=unlabeled_data, damping=args.damping)
    elif optimizer_class == GradientDescent:
        return GradientDescent(model.parameters(), lr=lr)


def train(model, train_data, optimizer, loss_function, tol, max_iter=float('inf'), print_every=10, name=None, folder_path=None):
    """Train the model until loss is minimized."""
    train_logs = {'condition': None, 'losses': []}
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

        # Gradient Norm Clipping
        #nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        #Gradient Value Clipping
        # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)

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
        # elif delta_loss > 1e-3 or np.isnan(delta_loss):
        #     condition = 'overshooting'
        #     model.load_state_dict(previous_model_state)  # recover previous model
        #     current_loss = previous_loss
        elif current_loss <= tol:
            condition = 'tol'
        elif epoch >= max_iter:
            condition = 'max_iter'

        train_logs['losses'].append(current_loss)

    # Final print
    print('*** FINAL EPOCH ***')
    print(f'Epoch {epoch}: Train loss: {current_loss:.4f}, Stop condition: {condition}')
    #
    # Save train logs
    train_logs['condition'] = condition

    # Return loss
    return current_loss, train_logs


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


def generate_linear_data(sigma2):
    global d, r2, ro, train_size, test_size, extra_size
    w_star = generate_true_parameter(d, r2, m=np.eye(d))
    c = generate_c(ro, regime='autoregressive', d=d)
    dataset = CenteredLinearGaussianDataset(w_star=w_star, d=d, c=c, n=train_size + test_size + extra_size, sigma2=sigma2)
    train_data, test_data, extra_data = random_split(dataset, [train_size, test_size, extra_size])

    return train_data, test_data, extra_data


def generate_quadratic_data(sigma2):
    global d, r2, ro, train_size, test_size, extra_size
    w_star = generate_true_parameter(d, r2, m=np.eye(d))
    W_star = generate_W_star(d, r2)
    c = generate_c(ro, regime='autoregressive', d=d)
    dataset = CenteredQuadraticGaussianDataset(
        W_star=W_star, w_star=w_star, d=d, c=c,
        n=train_size + test_size + extra_size, sigma2=sigma2)

    train_data, test_data, extra_data = random_split(dataset, [train_size, test_size, extra_size])

    return train_data, test_data, extra_data


def save_results(test_errors, folder_path, train_logs, name):
    mean_test_errors = defaultdict(list)
    for optim_cls, test_losses in test_errors.items():
        test_losses = np.array(test_losses)
        test_losses = test_losses.reshape(test_losses.shape[0], test_losses.shape[1])
        mean_test_errors[optim_cls] = np.nanmean(test_losses, axis=0).tolist()

    # Save test_errors
    with open(os.path.join(folder_path, 'test_errors.pkl'), 'wb') as f:
        pickle.dump(test_errors, f)
        # Save mean_test_errors
    with open(os.path.join(folder_path, 'mean_test_errors.pkl'), 'wb') as f:
        pickle.dump(mean_test_errors, f)
        # Save figure

    # Save train logs
    plt.title(name + ' | ' + train_logs['condition'])
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.plot(train_logs['losses'])

    plt.savefig(os.path.join(folder_path, 'train_logs', f'{name}.pdf'))
    plt.close()

    with open(os.path.join(folder_path, 'train_logs', f'{name}_train_logs.pkl'), 'wb') as f:
        pickle.dump(train_logs, f)

    # Plot the results
    try:
        for optim_cls in optimizer_classes:
            plt.scatter(noise_variances, mean_test_errors[optim_cls.__name__], label=optim_cls.__name__)

            plt.xlabel('Noise variance')
            plt.ylabel('Test loss')
            plt.legend([optim_cls.__name__ for optim_cls in optimizer_classes])

        plt.savefig(os.path.join(folder_path, 'plot.pdf'), format='pdf')
        plt.show()
    except:
        pass
    # Print path to results
    print(f'Results saved to {folder_path}')

# Fix parameters
tol = args.tol  # Eduard comment: This needs to be a lot smaller later on
lr = 1e-3
d = 10
num_layers = 3
loss_function = torch.nn.MSELoss()
r2 = 1  # signal
ro = 0.5
# Eduard comment: We are interested in cases where num_params > train_size (not just d > train_size)
# it is interesting that you found better generalization of NGD even if num_params <  train_size
width = args.width
max_iter = args.max_iter  # float('inf')
train_size = args.train_size
num_params = (1 + d) * width + width**2
extra_size = max(2*num_params - train_size, 10*train_size)
test_size = args.test_train_ratio*train_size

model = MLP(in_channels=d, num_layers=num_layers, hidden_channels=width).double().to(settings.DEVICE)
# Fix variables
noise_variances = np.linspace(1, 10, 20)
optimizer_classes = [GradientDescent, Kfac]

threshold = args.test_loss_threshold if args.test_loss_threshold else 0

if __name__ == '__main__':
    # if args.load_results and args.folder_path:
    #     folder_path = args.folder_path
    #     test_errors = pickle.load(open(os.path.join(folder_path, 'test_errors.pkl'), 'rb'))
    #     # replace values larger than threshold with nans
    #     for optim_cls, test_losses in test_errors.items():
    #         test_losses = np.array(test_losses)
    #         test_losses[test_losses > threshold] = np.nan
    #         test_errors[optim_cls] = test_losses.tolist()
    #     save_results(test_errors, folder_path)
    # else:
    # Create dictionary with all the parameters
    params_dict = {
        'tol': tol,
        'lr': lr,
        'extra_size': extra_size,
        'd': d,
        'num_params': num_params,
        'train_size': train_size,
        'test_size': test_size,
        'loss_function': str(loss_function),
        'num_runs': args.num_runs,
        'num_layers': num_layers,
        'max_iter': max_iter,
        'test_train_ratio': args.test_train_ratio,
        'r2': r2,
        'ro': ro,
    }

    # Create folder name
    dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    folder_name = dtstamp + '_' + '_'.join([f'{k}={v}' for k, v in params_dict.items()])
    folder_path = os.path.join(plots_dir, 'optimizer_benchmark', folder_name)
    os.makedirs(folder_path)
    os.makedirs(os.path.join(folder_path, 'train_logs'))

    # Save params
    with open(os.path.join(folder_path, 'params.json'), 'w') as f:
        json.dump(params_dict, f)


    test_errors = defaultdict(list)
    for run_id in range(args.num_runs):
        run_test_errors = defaultdict(list)
        for sigma2 in noise_variances:
            # Generate data
            train_data, test_data, extra_data = generate_quadratic_data(sigma2=sigma2)
            print(f'Noise variance: {sigma2}')
            for optim_cls in optimizer_classes:
                name = optim_cls.__name__ + f'_sigma2={sigma2}' + f'_run={1 + run_id}'
                print(f'\n\nOptimizer: {optim_cls.__name__}')
                if optim_cls.__name__ == 'Kfac':
                    mlp_output_sizes = ([width] * (num_layers - 1) if num_layers > 1 else []) + [1]
                    train_loss, hk_model, params, train_logs = kfac_train(train_data, mlp_output_sizes,
                                                                         max_iter=max_iter,
                                                              damping=args.damping, tol=tol, name=name,
                                                              folder_path=folder_path, print_every=args.print_every)
                    test_loss = kfac_test(hk_model, params, test_data)
                else:
                    # Instantiate the optimizer
                    optimizer = instantiate_optimizer(optim_cls, train_data, extra_data)
                    # Train the model
                    train_loss, train_logs = train(model, train_data, optimizer, loss_function, tol, max_iter,
                                       print_every=args.print_every, name=name, folder_path=folder_path)
                    # Test the model
                    test_loss = test(model, test_data, loss_function)
                    test_loss = test_loss if (test_loss < threshold or not threshold) else float('nan')
                    model.reset_parameters()

                run_test_errors[optim_cls.__name__].append(test_loss)

                for optim_cls, test_losses in run_test_errors.items():
                    test_errors[optim_cls].append(test_losses)

                # Save Results
                save_results(test_errors, folder_path, train_logs, name)

    # make dir, save plot and params





