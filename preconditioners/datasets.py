import torch
import torch.utils.data
from preconditioners.utils import generate_centered_gaussian_data


class CenteredGaussianDataset(torch.utils.data.Dataset):
    """
    Prepare the Packages dataset for regression
    """

    def __init__(self, w_star, c, n=200, d=600, sigma2=1, fix_norm_of_x=False):
        X, y, _ = generate_centered_gaussian_data(w_star, c, n, d, sigma2, fix_norm_of_x)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
