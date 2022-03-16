import unittest, torch

from torch import nn
from torch.utils.data import random_split

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
        self.train_dataset, self.extra_dataset = random_split(self.dataset, [train_size, extra_size])

        self.labeled_data = self.train_dataset[:][0].double().to(settings.DEVICE)
        self.unlabeled_data = self.extra_dataset[:][0].double().to(settings.DEVICE)

        self.model = SLP(in_channels=self.dataset.X.shape[1]).double().to(settings.DEVICE)
        self.optimizer = PrecondGD(self.model, lr=1e-3, labeled_data=self.labeled_data,
                                   unlabeled_data=self.unlabeled_data)

    def train(self, model, train_dataset, optimizer, loss_function, n_epochs, print_every=1):
        for epoch in range(n_epochs):
            model.train()
            # Set current loss value
            current_loss = 0.0

            # Get and prepare inputs
            inputs, targets = train_dataset[:]
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
            current_loss += loss.item()

            current_loss /= len(train_dataset)

            matrix_loss = np.linalg.norm(optimizer._compute_p_inv() - np.linalg.inv(self.c))

            if epoch % print_every == 0:
                print(f'Epoch {epoch + 1}: Train loss: {current_loss:.4f} Matrix error: {matrix_loss:.4f}')

    def test_p_inv_against_c_inv(self):
        self.train(
            self.model,
            self.train_dataset,
            self.optimizer,
            nn.MSELoss(),
            n_epochs=1)

        p_inv = self.optimizer._compute_p_inv()
        c_inv = np.linalg.inv(self.c)
        mat_error = np.linalg.norm(p_inv - c_inv, ord=np.inf) # max(sum(abs(p_inv - c_inv), axis=1))
        tol_error_in_each_entry = 0.3
        tol = p_inv.shape[0]*tol_error_in_each_entry
        self.assertTrue(
            mat_error < tol,
            msg=f"The error is {mat_error} and it should be less than {tol}, \
            the first entries of p_inv are {p_inv[:4,:4]}\
            while the first entries of c are {c_inv[:4,:4]}")

    def test_p_inv_against_true_p_inv(self):
        self.train(
                self.model,
                self.train_dataset,
                self.optimizer,
                nn.MSELoss(),
                n_epochs=1)

        p_inv = self.optimizer._compute_p_inv()

        true_p_inv = (1 / self.labeled_data.shape[0]) * self.labeled_data.T @ self.labeled_data
        true_p_inv += (1 / self.unlabeled_data.shape[0]) * self.unlabeled_data.T @ self.unlabeled_data
        true_p_inv = torch.inverse(true_p_inv)

        mat_error = torch.norm(p_inv - true_p_inv)

        self.assertTrue(
            mat_error < 0.01,
            msg=f"The error is {mat_error} and \
            the first entries of p_inv are {p_inv[:4,:4]}\
            while the first entries of X^TX/n + X'^TX'/n' are '{true_p_inv[:4,:4]}"
        )
