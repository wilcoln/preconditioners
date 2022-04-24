import argparse

from matplotlib import pyplot as plt
from torch.utils.data import random_split

from preconditioners import settings
from preconditioners.datasets import CenteredLinearGaussianDataset
from preconditioners.cov_approx.impl_cov_approx import *
from preconditioners.utils import generate_c, MLP
from preconditioners.optimizers import PrecondGD


class CheckEigenValues:
    def __init__(self, args):
        step = max(1, args.max_width//10)
        self.widths = np.arange(1, args.max_width + 1, step)
        self.d = 10
        self.train_size = 100
        self.w_star = np.random.multivariate_normal(mean=np.zeros(self.d), cov=np.eye(self.d))
        self.c = generate_c(ro=.5, regime='autoregressive', d=self.d)

    def setup(self, width):
        p = (1 + self.d) * width + width**2
        N = 3 * max(p, self.train_size)
        extra_size = N - self.train_size

        self.dataset = CenteredLinearGaussianDataset(w_star=self.w_star, d=self.d, c=self.c, n=N)
        self.train_dataset, extra_dataset = random_split(self.dataset, [self.train_size, extra_size])

        self.labeled_data = self.train_dataset[:][0].double().to(settings.DEVICE)
        self.unlabeled_data = extra_dataset[:][0].double().to(settings.DEVICE)
        model = MLP(in_channels=self.labeled_data.shape[1], num_layers=3, hidden_channels=width).double().to(
            settings.DEVICE)
        self.optimizer = PrecondGD(model, lr=1e-2, labeled_data=self.labeled_data, unlabeled_data=self.unlabeled_data)

    def run(self):
        min_eigen_values, max_eigen_values = [], []
        for width in self.widths:
            self.setup(width)
            F_inv = self.optimizer._compute_p_inv()
            grad = self.optimizer._compute_grad_of_data(self.labeled_data)
            m = grad @ F_inv @ grad.T
            eigen_values = np.linalg.eigvals(m)
            eig_min, eig_max = np.min(eigen_values), np.max(eigen_values)
            min_eigen_values.append(eig_min)
            max_eigen_values.append(eig_max)

        # Plot on a same graph min and max eigen values with respect to width
        plt.scatter(self.widths, min_eigen_values, label='min eigen values')
        plt.scatter(self.widths, max_eigen_values, label='max eigen values')
        plt.legend()
        plt.xlabel('Width')
        plt.ylabel('Eigen values')
        plt.show()


if __name__ == "__main__":
    # Get params
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_width', help='Max 3-MLP width', default=100, type=int)
    args = parser.parse_args()

    # Run
    check_eigen_values = CheckEigenValues(args)
    check_eigen_values.run()
