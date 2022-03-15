import unittest

from torch import nn
from torch.utils.data import random_split, DataLoader

from preconditioners import settings
from preconditioners.datasets import CenteredGaussianDataset
from preconditioners.cov_approx.impl_cov_approx import *
from preconditioners.utils import generate_c, SLP
from preconditioners.optimizers import PrecondGD


class TestPinv(unittest.TestCase):

    def setUp(self):
        d = 30
        train_size = 10
        extra_size = 1000
        w_star = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
        self.c = generate_c(ro=.5, regime='autoregressive', d=d)
        self.dataset = CenteredGaussianDataset(w_star=w_star, d=d, c=self.c, n=train_size + extra_size)
        train_dataset, extra_dataset = random_split(self.dataset, [train_size, extra_size])
        self.trainloader = DataLoader(train_dataset, batch_size=train_size, shuffle=True,
                                    num_workers=settings.NUM_WORKERS)

        labeled_data = train_dataset[:][0].to(settings.DEVICE)
        unlabeled_data = extra_dataset[:][0].to(settings.DEVICE)

        self.model = SLP(in_channels=self.dataset.X.shape[1])
        self.optimizer = PrecondGD(self.model, lr=1e-3, labeled_data=labeled_data, unlabeled_data=unlabeled_data)

    def train(self, model, trainloader, optimizer, loss_function, n_epochs, print_every=1):
        for epoch in range(n_epochs):
            model.train()
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
                outputs = model(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Update statistics
                current_loss += loss.item()

            current_loss /= len(trainloader)

            matrix_loss = np.linalg.norm(optimizer._compute_p_inv() - np.linalg.inv(self.c))

            if epoch % print_every == 0:
                print(f'Epoch {epoch + 1}: Train loss: {current_loss:.4f} Matrix error: {matrix_loss:.4f}')

    def test_p_inv(self):
        self.train(
            self.model,
            self.trainloader,
            self.optimizer,
            nn.MSELoss(),
            n_epochs=10)

        p_inv = self.optimizer._compute_p_inv()
        mat_error = np.linalg.norm(p_inv - np.linalg.inv(self.c))
        self.assertTrue(
            mat_error < 0.01,
            msg=f'The error is {mat_error} and \
            the first entries of p_inv are {p_inv[:4,:4]}\
            while the first entries of the true matrix are {np.linalg.inv(self.c)[:4,:4]}\
            and the first entries of X^TX/n are '
                f'{(self.dataset.X.T @ self.dataset.X/self.dataset.X.shape[0])[:4,:4]}'
        )
