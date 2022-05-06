"""Compares NGD training on an MLP with its linearization"""
import numpy as np
import torch
import argparse
import os
import pickle
import json
from datetime import datetime as dt

from torch.utils.data import random_split

from preconditioners import settings
from preconditioners.utils import generate_true_parameter, generate_c, SLP, MLP, LinearizedModel, generate_W_star
from preconditioners.datasets import CenteredLinearGaussianDataset, CenteredQuadraticGaussianDataset
from preconditioners.optimizers import GradientDescent, PrecondGD, PrecondGD2, PrecondGD3
from preconditioners.utils import model_gradients

parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', help='Number of layers', default=1, type=int)
parser.add_argument('--width', help='Hidden channels', type=int)
parser.add_argument('--max_iter', help='Max epochs', default=100, type=int)
parser.add_argument('--damping', help='damping', type=float, default=1.0)
parser.add_argument('--num_weights', help='Number of weights to analyse', type=float, default=100)
parser.add_argument('--save_folder', help='Experiments are saved here', type=str)
args = parser.parse_args()

def train_step(model, inputs, targets, optimizer, loss_function, bias=0):
    """Performs one step of training"""
    model.train()
    # Zero the gradients
    optimizer.zero_grad()

    # Perform the forward and backward pass
    outputs = model(inputs) + bias
    loss = loss_function(outputs, targets)
    loss.backward()

    optimizer.step()

    return loss

def train(model, model_lin, optimizer, optimizer_lin, train_data, loss_function, num_weights=10, max_iter=float('inf'), save_every=10):
    """Trains both models"""
    results = []
    epoch = 0

    # Get and prepare inputs
    inputs, targets = train_data[:]
    # Set the inputs and targets to the device
    inputs, targets = inputs.double().to(settings.DEVICE), targets.double().to(settings.DEVICE)
    targets = targets.reshape((targets.shape[0], 1))

    # TODO: this is computed twice (see bottom of page)
    ntk_features = model_gradients(model, inputs).detach()
    ntk_bias = model(inputs).detach()

    init_params = [p.clone().detach() for p in model.parameters()]

    # Choose random weights to analyse
    num_params = sum([np.prod(p.size()) for p in model_lin.parameters()])
    weight_indices = np.random.randint(0, num_params, size=num_weights)
    weight_indices = np.sort(weight_indices)

    print("Starting training...")
    print("Epoch\tLoss\tLinear Loss")

    # Sanity check
    outputs = model(inputs)
    loss = loss_function(outputs, targets)

    outputs = model_lin(ntk_features) + ntk_bias
    loss_lin = loss_function(outputs, targets)

    def save_result(save_param_norm=False):
        print(f'{epoch}\t{loss.item():.4f}\t{loss_lin.item():.4f}')

        # Compute frobenius norm of theta_t - theta_0
        frob_distance = 0
        for i, p in enumerate(model.parameters()):
            frob_distance += torch.square(torch.norm(p - init_params[i]))
        frob_distance = torch.sqrt(frob_distance)

        # Collect random weights
        flat_weights = unpack_weights(model.parameters())
        flat_weights_lin = unpack_weights(model_lin.parameters())
        random_weights = [flat_weights[i].item() for i in weight_indices]
        random_weights_lin = [flat_weights_lin[i].item() for i in weight_indices]

        result = {
            'epoch': epoch,
            'train_loss': loss.item(),
            'train_loss_lin': loss_lin.item(),
            'frob_distance': frob_distance.item(),
            'random_weights': random_weights,
            'random_weights_lin': random_weights_lin
        }

        if save_param_norm:
            # Compute frobenius norm of theta_t
            frob_norm = sum([torch.square(torch.norm(p)) for p in model.parameters()])
            result['param_norm'] = torch.sqrt(frob_norm).item()
        results.append(result)

    save_result(save_param_norm=True)

    while epoch < max_iter:
        loss = {'t':4 }
        loss = train_step(model, inputs, targets, optimizer, loss_function)
        loss_lin = train_step(model_lin, ntk_features, targets, optimizer_lin, loss_function, ntk_bias)

        epoch += 1

        # Print statistics
        if epoch == 1 or epoch % save_every == 0:
            save_result()

    # Final print
    print(f'{epoch}\t{loss.item():.4f}\t{loss_lin.item():.4f}')
    save_result()

    return results

def unpack_weights(params, use_cuda=False):
    tmp = torch.tensor([])
    for w in params:
        tmp = torch.cat((tmp, w.reshape(-1)))

    return tmp

def save_results(results, params_dict, folder_path):
    dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
    experiment_name = "training_comparison_" + dtstamp

    os.makedirs(os.path.join(folder_path, experiment_name))

    with open(os.path.join(folder_path, experiment_name, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)
    with open(os.path.join(folder_path, experiment_name, 'params.json'), 'w') as f:
        json.dump(params_dict, f)

def generate_quadratic_data(sigma2):
    """Generates test data. Taken from optimizer_benchmark"""
    global d, train_size, test_size, extra_size
    w_star = generate_true_parameter(d, 1, m=np.eye(d))
    W_star = generate_W_star(d, 1)
    c = generate_c(0.5, regime='autoregressive', d=d)
    dataset = CenteredQuadraticGaussianDataset(
        W_star=W_star, w_star=w_star, d=d, c=c,
        n=train_size + test_size + extra_size, sigma2=sigma2)

    train_data, test_data, extra_data = random_split(dataset, [train_size, test_size, extra_size])

    return train_data, test_data, extra_data

lr = 1e-2
max_iter = args.max_iter
d = 10

if __name__ == "__main__":
    # Generate data
    train_size, test_size, extra_size = 1000, 100, 10000
    train_data, test_data, extra_data = generate_quadratic_data(sigma2=1)

    # Create models
    width, num_layers = args.width, args.num_layers
    model = MLP(in_channels=d, num_layers=num_layers, hidden_channels=width)
    model_lin = LinearizedModel(model=model).double().to(settings.DEVICE)
    model = model.double().to(settings.DEVICE)
    loss_function = torch.nn.MSELoss()

    # Create optimizers
    labeled_data = train_data[:][0].double().to(settings.DEVICE)
    unlabeled_data = extra_data[:][0].double().to(settings.DEVICE)

    labeled_features = model_gradients(model, labeled_data).detach()
    unlabeled_features = model_gradients(model, unlabeled_data).detach()

    optimizer = PrecondGD(model, lr=lr, labeled_data=labeled_data, unlabeled_data=unlabeled_data, damping=args.damping)
    optimizer_lin = PrecondGD(model_lin, lr=lr, labeled_data=labeled_features, unlabeled_data=unlabeled_features, damping=args.damping, is_linear=True)

    # Train models
    results = train(model, model_lin, optimizer, optimizer_lin, train_data, loss_function, max_iter=args.max_iter, num_weights=args.num_weights)

    params_dict = {
        'extra_size': extra_size,
        'd': d,
        'train_size': train_size,
        'test_size': test_size,
        'loss_function': str(loss_function),
        'num_layers': num_layers,
        'max_iter': max_iter,
        'width': width
    }

    if args.save_folder:
        save_results(results, params_dict, args.save_folder)
