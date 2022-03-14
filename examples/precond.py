import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from preconditioners import settings
from preconditioners.datasets import CenteredGaussianDataset
from preconditioners.optimizers import PrecondGD

# Set fixed random number seed
torch.manual_seed(0)


# Create a dataset
dataset = CenteredGaussianDataset(w_star=np.ones(2), d=2, c=np.ones((2, 2)), n=1000)


# Create an MLP model
class MLP(nn.Module):
    """ Multilayer Perceptron for regression. """

    def __init__(self, in_size):
        super().__init__()
        self.layers = nn.Sequential(
            # nn.Linear(in_size, 1),
            nn.Linear(in_size, 2, bias=False),
            # nn.ReLU(),
            # nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1, bias=False)
        )

    def forward(self, x):
        """ Forward pass of the MLP. """
        return self.layers(x)


# Split dataset
train_size = int(0.7 * len(dataset))
test_size = int(0.2 * len(dataset))
extra_size = len(dataset) - train_size - test_size

train_dataset, test_dataset, extra_dataset = random_split(dataset, [train_size, test_size, extra_size])

# Create trainloader object
trainloader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_WORKERS)

# Create testloader object
testloader = DataLoader(test_dataset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_WORKERS)

# Initialize the MLP model
mlp = MLP(in_size=dataset.X.shape[1])

# Send the model to device
mlp.to(settings.DEVICE)

# Define the loss function
loss_function = nn.MSELoss()

# Define the optimizer
labeled_data = train_dataset[:][0].to(settings.DEVICE)
unlabeled_data = extra_dataset[:][0].to(settings.DEVICE)
optimizer = PrecondGD(mlp, lr=1e-4, labeled_data=labeled_data, unlabeled_data=unlabeled_data)


losses = {'train': [], 'test': []}


def train(epoch, silent=False):

    mlp.train()

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(trainloader, 0):

        # Get and prepare inputs
        inputs, targets = data
        # Set the inputs and targets to the device
        inputs, targets = inputs.to(settings.DEVICE), targets.to(settings.DEVICE)

        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Update statistics
        current_loss += loss.item()

    current_loss /= len(trainloader)

    losses['train'].append(current_loss)

    if not silent:
        print(f'Epoch {epoch + 1}: Train loss: {current_loss:.4f}')


def test(epoch, silent=False):
    mlp.eval()

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for i, data in enumerate(testloader, 0):

        # Get and prepare inputs
        inputs, targets = data
        inputs, targets = inputs.float(), targets.float()
        targets = targets.reshape((targets.shape[0], 1))

        # Perform forward pass
        outputs = mlp(inputs)

        # Compute loss
        loss = loss_function(outputs, targets)

        # Print statistics
        current_loss += loss.item()

    current_loss /= len(testloader)

    losses['test'].append(current_loss)

    if not silent:
        print(f'Epoch {epoch + 1}: Test loss: {current_loss:.4f}')


# Run the training loop
for epoch in range(2):
    train(epoch)
    test(epoch)

# from torch.autograd import grad
#
# x = torch.ones(2, 2, requires_grad=True)
#
# y = (x+1) ** 2
#
# print(x)
# print(y)
# dy_dx = grad(outputs=y, inputs=x, grad_outputs=torch.ones_like(y))
# print(f'dy/dx:\n {dy_dx}')
