import torch
import unittest
from torch.utils.data import random_split

from preconditioners import settings
from preconditioners.cov_approx.impl_cov_approx import *
from preconditioners.datasets import CenteredLinearGaussianDataset, generate_c, generate_true_parameter
from preconditioners.optimizers import PrecondGD, PrecondGD3
from preconditioners.utils import SLP
from datetime import datetime as dt

import numpy as np
import torch
from matplotlib import pyplot as plt

class TestPinv(unittest.TestCase):

    def setUp(self):
        d = 50
        train_size = 20
        extra_size = 10000
        ro = 0.8
        w_star = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
        self.c = generate_c(ro=ro, regime='autoregressive', d=d)
        self.dataset = CenteredLinearGaussianDataset(w_star=w_star, d=d, c=self.c, n=train_size + extra_size)
        self.train_dataset, self.extra_dataset = random_split(self.dataset, [train_size, extra_size])

        self.labeled_data = self.train_dataset[:][0].double().to(settings.DEVICE)
        self.unlabeled_data = self.extra_dataset[:][0].double().to(settings.DEVICE)

        self.model = SLP(in_channels=self.dataset.X.shape[1]).double().to(settings.DEVICE)
        self.optimizer = PrecondGD(self.model, lr=1e-3, labeled_data=self.labeled_data,
                                   unlabeled_data=self.unlabeled_data, damping=0.)

    def test_p_inv_against_c_inv(self):

        p_inv = self.optimizer._compute_p_inv()
        c_inv = np.linalg.inv(self.c)
        mat_error = np.linalg.norm(p_inv - c_inv, ord=np.inf)  # max(sum(abs(p_inv - c_inv), axis=1))
        tol_error_in_each_entry = 0.1
        tol = p_inv.shape[0]*tol_error_in_each_entry
        print(mat_error)
        self.assertTrue(
            mat_error < tol,
            msg=f"The error is {mat_error} and it should be less than {tol}, \
            the first entries of p_inv are {p_inv[:4,:4]}\
            while the first entries of c are {c_inv[:4,:4]}")

    def test_p_inv_against_true_p_inv(self):

        p_inv = self.optimizer._compute_p_inv()

        #true_p_inv = (1 / self.labeled_data.shape[0]) * self.labeled_data.T @ self.labeled_data
        #true_p_inv += (1 / self.unlabeled_data.shape[0]) * self.unlabeled_data.T @ self.unlabeled_data
        # true_p_inv = self.labeled_data.T @ self.labeled_data + self.unlabeled_data.T @ self.unlabeled_data
        # true_p_inv *= 1/(self.labeled_data.shape[0] + self.unlabeled_data.shape[0])
        true_p_inv = self.unlabeled_data.T @ self.unlabeled_data
        true_p_inv *= 1/self.unlabeled_data.shape[0]
        true_p_inv = torch.cholesky_inverse(torch.linalg.cholesky(true_p_inv))

        mat_error = torch.norm(p_inv - true_p_inv)

        self.assertTrue(
            mat_error < 0.01,
            msg=f"The error is {mat_error} and \
            the first entries of p_inv are {p_inv[:4,:4]}\
            while the first entries of X^TX/n + X'^TX'/n' are '{true_p_inv[:4,:4]}"
        )


    def test_p_inv_with_damping(self):

        N = 20
        extra_size = 10000
        sigma2 = 2
        r2=1
        ro = 0.9
        d = 50
        lam = 0.1
        damping = lam * np.sqrt(d)
        np.random.seed(0)

        c = np.sqrt(d) * generate_c(ro=ro, regime='autoregressive', d=d)
        X = np.random.multivariate_normal(mean=np.zeros(d), cov=c, size=N)
        P_damped = np.linalg.inv(c + damping*np.eye(d))

        w_star = generate_true_parameter(d, r2=r2, m=np.eye(d))
        dataset = CenteredLinearGaussianDataset(w_star, c, n=N+extra_size, d=d, sigma2=sigma2)

        train_dataset, extra_dataset = random_split(dataset, [N, extra_size])
        labeled_data = train_dataset[:][0].double().to(settings.DEVICE)
        unlabeled_data = extra_dataset[:][0].double().to(settings.DEVICE)

        model = SLP(in_channels=d).double().to(settings.DEVICE)
        optimizer = PrecondGD(model, lr=1e-2, labeled_data=labeled_data, unlabeled_data=unlabeled_data, verbose=False, damping=damping)

        F_inv = optimizer._compute_p_inv()

        tol = 0.1
        self.assertTrue(np.linalg.norm(F_inv - P_damped)/np.sqrt(F_inv.shape[0])<0.1, 
                        msg=f"Error is {np.linalg.norm(F_inv - P_damped)/np.sqrt(F_inv.shape[0])} while tol is {tol}")
