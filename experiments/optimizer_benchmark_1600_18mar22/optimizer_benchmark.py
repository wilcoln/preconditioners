""" Compare different optimizers and plot the results."""
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import random_split
import matplotlib.pyplot as plt

from icecream import ic

from preconditioners import settings
from preconditioners.datasets import CenteredGaussianDataset
from preconditioners.optimizers import GradientDescent, PrecondGD
# Eduard comment: The way we will test it, these are going to be the sizes:
# len(test_data) < len(train_data) < no_of_parameters_of_the_model << len(extra_data)
from utils import generate_c, MLP


# Helper functions
def instantiate_optimizer(optimizer_class, train_data, extra_data):
    if optimizer_class == PrecondGD:
        labeled_data = train_data[:][0].double().to(settings.DEVICE)
        unlabeled_data = extra_data[:][0].double().to(settings.DEVICE)
        return PrecondGD(model, lr=lr, labeled_data=labeled_data, unlabeled_data=unlabeled_data)
    elif optimizer_class == GradientDescent:
        return GradientDescent(model.parameters(), lr=lr)


def train(model, train_data, optimizer, loss_function, tol, max_iter=float('inf'), print_every=10):
    """Train the model until loss is minimized."""
    model.train()
    current_loss = float('inf')
    epoch = 0
    # stop if 5 consecutive epochs have no improvement
    no_improvement_counter = 0

    while current_loss > tol and epoch < max_iter and no_improvement_counter < 5:
        previous_loss = current_loss

        model.train()

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

        if current_loss >= previous_loss:
            no_improvement_counter += 1
        else:
            no_improvement_counter = 0

        epoch += 1

        if epoch % print_every == 0:
            print(f'Epoch {epoch}: Train loss: {current_loss:.4f}')

    return current_loss


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


def generate_data(ro):
    global d, train_size, test_size, extra_size
    w_star = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
    c = generate_c(ro=ro, regime='autoregressive', d=d)
    dataset = CenteredGaussianDataset(w_star=w_star, d=d, c=c, n=train_size + test_size + extra_size)
    train_data, test_data, extra_data = random_split(dataset, [train_size, test_size, extra_size])

    return train_data, test_data, extra_data


# Fix parameters
tol = 1e-2
lr = 1e-1
extra_size = 1000
d = 4
num_params = int(.1 * extra_size)
train_size = int(.5 * num_params)
test_size = int(.5 * train_size)
loss_function = torch.nn.MSELoss()
model = MLP(in_channels=d, num_layers=2, hidden_layer_size=num_params//(1 + d)).double().to(settings.DEVICE)
max_iter = 1000  # float('inf')


# Fix variables
noise_variances = np.linspace(0, .5, 10)
optimizer_classes = [GradientDescent, PrecondGD]


test_errors = defaultdict(list)
for ro in noise_variances:
    # Generate data
    train_data, test_data, extra_data = generate_data(ro=ro)
    print(f'Noise variance: {ro}')
    for optim_cls in optimizer_classes:
        print(f'\n\nOptimizer: {optim_cls.__name__}')
        # Instantiate the optimizer
        optimizer = instantiate_optimizer(optim_cls, train_data, extra_data)
        # Train the model
        train_loss = train(model, train_data, optimizer, loss_function, tol, max_iter, print_every=10)
        # Test the model
        test_loss = test(model, test_data, loss_function)
        test_errors[optim_cls.__name__].append(test_loss)
        model.reset_parameters()

# Plot the results
for optim_cls in optimizer_classes:
    plt.plot(noise_variances, test_errors[optim_cls.__name__], label=optim_cls.__name__)

plt.xlabel('Noise variance')
plt.ylabel('Test loss')
plt.legend([optim_cls.__name__ for optim_cls in optimizer_classes])

plt.show()
