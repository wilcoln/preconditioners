import unittest

from torch import nn
from torch.utils.data import random_split

import settings
from preconditioners.datasets import CenteredGaussianDataset
from preconditioners.cov_approx.impl_cov_approx import *
from preconditioners.utils import generate_c, generate_centered_gaussian_data
from preconditioners.optimizers import PrecondGD


class TestPinv(unittest.TestCase):

    def test_p_inv(self):

        d = 30
        c = generate_c(ro=.5, regime='autoregressive', d=d)

        w_star = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
        dataset = CenteredGaussianDataset(w_star=w_star, d=d, c=c, n=1000)
        
        train_size = int(0.7 * len(dataset))
        test_size = int(0.2 * len(dataset))
        extra_size = len(dataset) - train_size - test_size

        train_dataset, test_dataset, extra_dataset = random_split(dataset, [train_size, test_size, extra_size])

        labeled_data = train_dataset[:][0].to(settings.DEVICE)
        unlabeled_data = extra_dataset[:][0].to(settings.DEVICE)

        class SLP(nn.Module):
            """ Multilayer Perceptron for regression. """

            def __init__(self, in_size):
                super().__init__()
                self.layers = nn.Sequential(nn.Linear(in_size, 1, bias=False))

            def forward(self, x):
                """ Forward pass of the MLP. """
                return self.layers(x)

        mlp = SLP(in_size=dataset.X.shape[1])
        optimizer = PrecondGD(mlp, lr=1e-3, labeled_data=labeled_data, unlabeled_data=unlabeled_data)

        p_inv = optimizer._compute_p_inv()
        
        self.assertTrue(
            np.linalg.norm(p_inv - np.linalg.inv(c)) < 0.01,
            msg=f'The error is {np.linalg.norm(p_inv - np.linalg.inv(c))} and \
            the first entries of p_inv are {p_inv[:4,:4]}\
            while the first entries of the true matrix are {np.linalg.inv(c)[:4,:4]}'
        )
