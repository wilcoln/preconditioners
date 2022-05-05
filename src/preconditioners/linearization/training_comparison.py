"""Compares NGD training on an MLP with its linearization"""
import numpy as np
import torch
import argparse

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
args = parser.parse_args()

def train_step(model, inputs, targets, optimizer, loss_function, bias=0):
    """Performs one step of training"""
    # Zero the gradients
    optimizer.zero_grad()

    # Perform the forward and backward pass
    outputs = model(inputs) + bias
    loss = loss_function(outputs, targets)
    loss.backward()

    optimizer.step()

    return loss

def train(model, model_lin, optimizer, optimizer_lin, train_data, loss_function, max_iter=float('inf'), print_every=10):
    """Trains both models"""
    # Get and prepare inputs
    inputs, targets = train_data[:]
    # Set the inputs and targets to the device
    inputs, targets = inputs.double().to(settings.DEVICE), targets.double().to(settings.DEVICE)
    targets = targets.reshape((targets.shape[0], 1))

    # TODO: this is computed twice (see bottom of page)
    ntk_features = model_gradients(model, inputs).detach()
    ntk_bias = model(inputs).detach()

    print("Starting training...")
    print("Epoch\tLoss\tLinear Loss")

    # Sanity check
    outputs = model(inputs)
    loss = loss_function(outputs, targets)

    outputs = model_lin(ntk_features) + ntk_bias
    loss_lin = loss_function(outputs, targets)
    print(f'0\t{loss.item():.4f}\t{loss_lin.item():.4f}')

    epoch = 0
    while True:
        model.train()
        loss = train_step(model, inputs, targets, optimizer, loss_function)
        model.train(False)
        loss_lin = train_step(model_lin, ntk_features, targets, optimizer_lin, loss_function, ntk_bias)

        epoch += 1

        # Print statistics
        if epoch == 1 or epoch % print_every == 0:
            print(f'{epoch}\t{loss.item():.4f}\t{loss_lin.item():.4f}')

        if epoch >= max_iter:
            break

    # Final print
    print(f'{epoch}\t{loss.item():.4f}\t{loss_lin.item()}')

    return loss.item()

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

lr = 1e-3
max_iter = args.max_iter
d = 10

if __name__ == "__main__":
    # Generate data
    train_size, test_size, extra_size = 1000, 100, 100
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
    train(model, model_lin, optimizer, optimizer_lin, train_data, loss_function)
