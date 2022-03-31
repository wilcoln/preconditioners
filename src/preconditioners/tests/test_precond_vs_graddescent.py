import unittest

from torch import nn
from torch.utils.data import random_split

from preconditioners import settings
from preconditioners.datasets import CenteredLinearGaussianDataset
from preconditioners.cov_approx.impl_cov_approx import *
from preconditioners.utils import generate_c, SLP
from preconditioners.optimizers import PrecondGD, GradientDescent


class TestPrecondVsGradDescent(unittest.TestCase):

    def setUp(self):
        d = 10
        train_size = 100
        extra_size = 900
        w_star = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
        self.c = generate_c(ro=.5, regime='autoregressive', d=d)
        self.dataset = CenteredLinearGaussianDataset(w_star=w_star, d=d, c=self.c, n=train_size + extra_size)
        self.train_dataset, extra_dataset = random_split(self.dataset, [train_size, extra_size])

        labeled_data = self.train_dataset[:][0].double().to(settings.DEVICE)
        unlabeled_data = extra_dataset[:][0].double().to(settings.DEVICE)

        self.model = SLP(in_channels=self.dataset.X.shape[1]).double().to(settings.DEVICE)
        self.optimizer1 = PrecondGD(self.model, lr=1e-1, labeled_data=labeled_data, unlabeled_data=unlabeled_data)
        self.optimizer2 = GradientDescent(self.model.parameters(), lr=1e-1)

    def train(self, model, train_dataset, optimizer, loss_function, n_epochs, print_every=1):
        for epoch in range(n_epochs):
            model.train()

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
            current_loss = loss.item()

            if epoch % print_every == 0:
                print(f'Epoch {epoch + 1}: Train loss: {current_loss:.4f}')

        return current_loss

    def test_precond_vs_graddescent(self):
        loss1 = self.train(
            self.model,
            self.train_dataset,
            self.optimizer1,
            nn.MSELoss(),
            n_epochs=10)

        self.model.reset_parameters()

        loss2 = self.train(
            self.model,
            self.train_dataset,
            self.optimizer2,
            nn.MSELoss(),
            n_epochs=10)

        self.assertLessEqual(loss1, loss2)
