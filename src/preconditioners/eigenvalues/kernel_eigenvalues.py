"""Computes eigenvalues of the preconditioned kernel over multiple n. Uses the
empirical Fisher information for preconditioning

Example:
python src/preconditioners/eigenvalues/kernel_eigenvalues.py --max_width 12 --ntk
"""
import argparse
import os
from datetime import datetime as dt
import pickle
import json
import numpy as np

from matplotlib import pyplot as plt
from torch.utils.data import random_split

from preconditioners import settings
from preconditioners.datasets import generate_true_parameter, CenteredQuadraticGaussianDataset, generate_W_star, generate_c
from preconditioners.utils import MLP
from preconditioners.optimizers import PrecondGD

def create_argparser():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--num_layers', help='Number of layers', type=int, default=3)
    parser.add_argument('--width_min', help='Hidden channels', type=int, default=4)
    parser.add_argument('--width_max', help='Hidden channels', type=int, default=64)
    parser.add_argument('--width_step', help='Hidden channels', type=int, default=4)
    # Optimizer params
    parser.add_argument('--damping', help='Damping coefficient', type=float, default=1e-1)
    # Data params
    parser.add_argument('--dataset', help='Type of dataset', choices=['linear', 'quadratic'], default='quadratic')
    parser.add_argument('--train_size', help='Number of train examples', type=int, default=32)
    parser.add_argument('--extra_size', help='Number of extra examples', type=int, default=256)
    parser.add_argument('--in_dim', help='Dimension of features', type=int, default=8)
    parser.add_argument('--ro', type=float, default=.5)
    parser.add_argument('--r2', type=float, default=1)
    parser.add_argument('--regime', type=str, default='autoregressive')
    parser.add_argument('--sigma2', help='Standard deviation of label noise', type=float, default=1)
    # Experiment params
    parser.add_argument('--save_folder', help='Experiments are saved here', type=str, default="experiments/")
    return parser.parse_args()

class CheckEigenvalues:
    def __init__(self, args):
        # Network parameters
        self.widths = np.arange(args.width_min, args.width_max + 1, args.width_step)
        self.d = args.in_dim
        self.damping = args.damping

        self.results = []
        self.save_folder = args.save_folder

        # Dataset parameters
        self.train_size = args.train_size
        self.extra_size = args.extra_size

        # Generate data
        w_star = generate_true_parameter(self.d, args.r2, m=np.eye(self.d))
        W_star = generate_W_star(self.d, args.r2)
        c = generate_c(args.ro, regime=args.regime, d=self.d)
        dataset = CenteredQuadraticGaussianDataset(
            W_star=W_star, w_star=w_star, d=self.d, c=c,
            n=self.train_size + self.extra_size, sigma2=args.sigma2)

        self.train_dataset, self.extra_dataset = random_split(dataset, [self.train_size, self.extra_size])
        self.labeled_data = self.train_dataset[:][0].double().to(settings.DEVICE)
        self.unlabeled_data = self.extra_dataset[:][0].double().to(settings.DEVICE)

    def create_model(self, width):
        std = 1/ np.sqrt(width)

        # TODO: isn't this wrong?
        # We have std = 1/sqrt(width) here, but inside MLP we initialize the weights with 
        # sigma_w = std
        # tmp = sigma_w / np.sqrt(self.hidden_channels)
        # layer.weight.data.normal_(0, tmp)
        # so this adds up to normal distribution with std = 1/sqrt(width) * sqrt(width)!!!!

        # Create model and optimizer
        model = MLP(in_channels=self.d, num_layers=3, hidden_channels=width, std=std).double().to(
            settings.DEVICE)
        self.optimizer = PrecondGD(model, lr=1e-2, labeled_data=self.labeled_data, unlabeled_data=self.unlabeled_data, verbose=False, damping=self.damping)

    def run(self):
        def log_result(width, eigenvalues):
            result = {
                'width': width,
                'eigenvalues': eigenvalues
            }

            self.results.append(result)

        for width in self.widths:
            print(f"Running experiment for n={width}")
            # Set up experiment
            self.create_model(width)

            # Compute Fisher information
            F_inv = self.optimizer._compute_p_inv()
            # Compute kernel
            grad = self.optimizer._compute_grad_of_data(self.labeled_data)
            m = grad @ F_inv @ grad.T

            #Compute eigenavalues
            eigenvalues = np.linalg.eigvals(m)
            log_result(width, eigenvalues)

    def save_results(self, params_dict):
        dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        experiment_name = "kernel_eigenvalues_" + dtstamp

        os.makedirs(os.path.join(self.save_folder, experiment_name))

        with open(os.path.join(self.save_folder, experiment_name, 'results.pkl'), 'wb') as f:
            pickle.dump(self.results, f)
        with open(os.path.join(self.save_folder, experiment_name, 'params.json'), 'w') as f:
            json.dump(vars(params_dict), f)

if __name__ == "__main__":
    args = create_argparser()

    # Run
    check_eigenvalues = CheckEigenvalues(args)
    check_eigenvalues.run()
    print("Finished!")
    check_eigenvalues.save_results(args)
