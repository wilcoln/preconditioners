import argparse
import os
from datetime import datetime as dt
import pickle
import json

from matplotlib import pyplot as plt
from torch.utils.data import random_split

from preconditioners import settings
from preconditioners.datasets import CenteredLinearGaussianDataset
from preconditioners.cov_approx.impl_cov_approx import *
from preconditioners.utils import generate_c, MLP
from preconditioners.optimizers import PrecondGD


class CheckEigenValues:
    def __init__(self, args):
        # Network parameters
        self.widths = np.arange(args.min_width, args.max_width + 1, args.width_step)
        self.d = 5
        self.use_ntk = args.ntk

        # Dataset parameters
        self.train_size = 50
        self.extra_size = 1000
        self.w_star = np.random.multivariate_normal(mean=np.zeros(self.d), cov=np.eye(self.d))
        self.c = generate_c(ro=.5, regime='autoregressive', d=self.d)

        self.save_folder = args.save_folder

    def create_dataset(self):
        N = self.train_size + self.extra_size
        self.dataset = CenteredLinearGaussianDataset(w_star=self.w_star, d=self.d, c=self.c, n=N)
        self.train_dataset, self.extra_dataset = random_split(self.dataset, [self.train_size, self.extra_size])

        self.labeled_data = self.train_dataset[:][0].double().to(settings.DEVICE)
        self.unlabeled_data = self.extra_dataset[:][0].double().to(settings.DEVICE)

    def create_model(self, width, use_ntk=False):
        std = 1
        if use_ntk:
            std /= np.sqrt(width)

        # Create model and optimizer
        model = MLP(in_channels=self.labeled_data.shape[1], num_layers=3, hidden_channels=width, std=std).double().to(
            settings.DEVICE)
        self.optimizer = PrecondGD(model, lr=1e-2, labeled_data=self.labeled_data, unlabeled_data=self.unlabeled_data, verbose=False)

    def run(self):
        results = []
        self.create_dataset()

        def log_result(width, eigenvalues):
            result = {
                'width': width,
                'eigenvalues': eigenvalues
            }

            results.append(result)

        for width in self.widths:
            print(f"Running experiment for n={width}")
            # Set up experiment
            self.create_model(width, self.use_ntk)

            # Compute Fisher information
            F_inv = self.optimizer._compute_p_inv()
            # Compute kernel
            grad = self.optimizer._compute_grad_of_data(self.labeled_data)
            m = grad @ F_inv @ grad.T

            #Compute eigenavalues
            eigenvalues = np.linalg.eigvals(m)
            log_result(width, eigenvalues)
        print("Finished!")

        self.save_results(results)
        print("Saved results!")

    def save_results(self, results):
        dtstamp = str(dt.now()).replace(' ', '_').replace(':', '-').replace('.', '_')
        experiment_name = "check_eigenvalues_" + dtstamp

        params_dict = {
            'use_ntk': self.use_ntk
        }

        os.makedirs(os.path.join(self.save_folder, experiment_name))

        with open(os.path.join(self.save_folder, experiment_name, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(self.save_folder, experiment_name, 'params.json'), 'w') as f:
            json.dump(params_dict, f)

if __name__ == "__main__":
    # Get params
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_width', help='Min 3-MLP width', default=4, type=int)
    parser.add_argument('--max_width', help='Max 3-MLP width', default=64, type=int)
    parser.add_argument('--width_step', help='Width step', default=4, type=int)
    parser.add_argument('--ntk', action='store_true', help='Use NTK parameterisation')
    parser.add_argument('--save_folder', help='Experiments are saved here', type=str, default="experiments/")

    args = parser.parse_args()

    # Run
    check_eigen_values = CheckEigenValues(args)
    check_eigen_values.run()
