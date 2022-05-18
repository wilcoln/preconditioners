import torch
import torch.utils.data
from icecream import ic
import numpy as np
import warnings
from preconditioners.utils import MLP

class NumpyDataset(torch.utils.data.Dataset):

    def __init__(self, x, y):
        self.X = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class CenteredLinearGaussianDataset(torch.utils.data.Dataset):
    """
    Prepare the Packages dataset for regression
    """

    def __init__(self, w_star, c, n=200, d=600, sigma2=1, fix_norm_of_x=False):
        X, y, _ = generate_centered_linear_gaussian_data(w_star, c, n, d, sigma2, fix_norm_of_x)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class CenteredQuadraticGaussianDataset(torch.utils.data.Dataset):
    """Prepare the Packages dataset for regression

    See 'generate_centered_quadratic_gaussian_data()' for more info
    """

    def __init__(self, W_star, w_star, c, n=200, d=600, sigma2=1, rng=None):
        X, y, _ = generate_centered_quadratic_gaussian_data(W_star, w_star, c, n, d, sigma2, rng=rng)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

class DataGenerator:

    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name
        self.kwargs = kwargs
        if dataset_name == 'linear':
            self.w_star = generate_true_parameter(**kwargs)
            self.c = generate_c(**kwargs)
        elif dataset_name == 'quadratic':
            self.w_star = generate_true_parameter(**kwargs)
            self.W_star = generate_W_star(**kwargs)
            self.c = generate_c(**kwargs)
        elif dataset_name == 'MLP':
            self.c = generate_c(**kwargs)
            self.model = MLP(in_channels=kwargs['d'], num_layers=kwargs['num_layers'], hidden_channels=kwargs['hidden_channels'])
        else:
            raise Error("Do not recognise dataset name")

    def generate(self, n, sigma2=None):
        if sigma2 is None:
            sigma2 = self.kwargs['sigma2']
        kwargs = self.kwargs.copy()
        kwargs['sigma2'] = sigma2

        if self.dataset_name == 'linear':
            x, y, _ = generate_centered_linear_gaussian_data(n=n, w_star=self.w_star, c=self.c, **self.kwargs)
        elif self.dataset_name == 'quadratic':
            x, y, _ = generate_centered_quadratic_gaussian_data(n=n, W_star=self.W_star, w_star=self.w_star, c=self.c, **self.kwargs)
        elif self.dataset_name == 'MLP':
            c = generate_c(**kwargs)
            x = np.random.multivariate_normal(mean=np.zeros(kwargs['d']), cov=self.c, size=n)
            y_noiseless = self.model(torch.from_numpy(x).float())
            y_noiseless = y_noiseless.cpu().detach().numpy()
            y_noiseless = np.squeeze(y_noiseless)
            y = y_noiseless + np.random.normal(0, sigma2, size=n)

        return x, y


def generate_data(dataset_name='linear', **kwargs):
    """Generates data

    Arguments
    n: number of datapoints
    d: dimension of feaures
    ro: coefficient in autoregressive regime
    r1: std of linear weights
    r2: std of quadratic weights
    m: covariance of linear weights (default: identity)
    sigma2: std of label noise
    fix_norm_of_x: TODO
    regime: regime for feature covariance
    strong_feature: TODO
    strong_feature_ratio: TODO
    weak_feature: TODO
    """
    # TODO: fix this so that **kwargs isn't needed in the other functions
    if dataset_name == 'linear':
        w_star = generate_true_parameter(**kwargs)
        c = generate_c(**kwargs)
        x, y, _ = generate_centered_linear_gaussian_data(w_star=w_star, c=c, **kwargs)
        return (x, y)
    elif dataset_name == 'quadratic':
        w_star = generate_true_parameter(**kwargs)
        W_star = generate_W_star(**kwargs)
        c = generate_c(**kwargs)
        x, y, _ = generate_centered_quadratic_gaussian_data(W_star=W_star, w_star=w_star, c=c, **kwargs)
        return (x, y)
    elif dataset_name == 'MLP':
        d, n, sigma2 = kwargs['d'], kwargs['n'], kwargs['sigma2']
        c = generate_c(**kwargs)
        X = np.random.multivariate_normal(mean=np.zeros(d), cov=c, size=n)
        model = MLP(in_channels=d, num_layers=kwargs['num_layers'], hidden_channels=kwargs['hidden_channels'])
        y_noiseless = model(torch.from_numpy(X).float()).cpu().detach().numpy()
        y = y_noiseless + np.random.normal(0, sigma2, size=(n, 1))
        y = np.squeeze(y)
        return (X, y)

    raise Exception(f"Unrecognised dataset name")

def generate_centered_linear_gaussian_data(w_star, c, n=200, d=10, sigma2=1, fix_norm_of_x=False, **kwargs):
    """Generates linear Gaussian data

    y = w*^T X + xi
    X: mean zero Gaussian with covariance c
    xi: mean zero Gaussian with std sigma2
    """
    assert len(w_star) == d, 'dimensions error'

    # generate features
    X = np.zeros((n, d))

    X = np.random.multivariate_normal(mean=np.zeros(d), cov=c, size=n)

    if fix_norm_of_x:
        X = X * np.sqrt(d) / np.linalg.norm(X, axis=1)[:, None]

    # print warning if X is not on the sphere
    if any(abs(np.linalg.norm(X, axis=1) - np.sqrt(d)) > 1e-5):
        warnings.warn('Warning, norms of datapoints are not sqrt(d)')

    # generate_noise
    xi = np.random.normal(0, np.sqrt(sigma2), size=n)

    # generate response
    y = X.dot(w_star) + xi

    return X, y, xi


def generate_centered_quadratic_gaussian_data(W_star, w_star, c, n=200, d=10, sigma2=1, rng=None, **kwargs):
    """Generates quadratic Gaussian data

    y = X^T W* X + X^T w* + xi
    X: mean zero Gaussian with covariance c
    xi: mean zero Gaussian with std sigma2
    """
    assert W_star.shape == (d, d), 'dimensions error'

    rng = rng if rng is not None else np.random

    # generate features
    X = rng.multivariate_normal(mean=np.zeros(d), cov=c, size=n)

    # generate_noise
    xi = rng.normal(0, np.sqrt(sigma2), size=n)

    # generate response
    y = (X.dot(W_star)*X).sum(axis=1) + X.dot(w_star) + xi # equivalent to np.array([X[i].T.dot(W_star.dot(X[i])) for i in range(n)]) + X.dot(w_star) + xi

    return X, y, xi

def generate_m(c, source_condition='id'):
    """Generates m for the linear dataset"""
    if source_condition == 'id':
        m = np.eye(c.shape[0])

    elif source_condition == 'easy':
        m = c

    elif source_condition == 'hard':
        m = np.linalg.inv(c)

    return m


def generate_true_parameter(d=10, r1=5, m=None, rng=None, **kwargs):
    """Generates w_star for the linear and quadratic dataset"""
    if m is None:
        m = np.eye(d)

    assert (m.shape[0] == d) & (m.shape[1] == d)

    rng = rng if rng is not None else np.random
    w_star = rng.multivariate_normal(np.zeros(d), r1 / d * m)

    return w_star

def generate_W_star(d=10, r2=1, rng=None, **kwargs):
    """Generates W_star for the quadratic model

    Return a somewhat random matrix of size (d, d), which does not have degenerate trace or determinant.
    """
    rng = rng if rng is not None else np.random
    ro = 0.4 + rng.rand()/2
    c = generate_c(ro=ro, regime='autoregressive', d=d)
    V, D, Vt = np.linalg.svd(c)

    for i in range(len(D)):
        if rng.rand()<0.25:
            D[i] = D[i]*0.5
        elif rng.rand()>0.75:
            D[i] = D[i]*2
    D = np.abs(D + rng.multivariate_normal(np.zeros(d), np.diag(D)/10))
    largest_eval = np.max(D)
    D = D/largest_eval

    W = V.dot(np.diag(D).dot(Vt))*np.sqrt(r2/d) # this way x^T W x ~ sqrt(r2/d), which is the same as x^T w_star
    return W


def generate_c(ro=0.5, regime='id', d=10, strong_feature=1, strong_feature_ratio=1 / 2, weak_feature=1, **kwargs):
    """Generates c for quadratic and linear dataset"""
    c = np.eye(d)

    if regime == 'id':
        pass

    elif regime == 'autoregressive':

        for i in range(d):
            for j in range(d):
                c[i, j] = ro ** (abs(i - j))

    elif regime == 'strong_weak_features':

        s_1 = np.ones(int(d * strong_feature_ratio)) * strong_feature
        s_2 = np.ones(d - int(d * strong_feature_ratio)) * weak_feature

        c = np.diag(np.concatenate((s_1, s_2)))

    elif regime == 'exponential':

        s = np.linspace(0, 1, d + 1, endpoint=False)[1:]
        quantile = - np.log(1 - s)  # quantile function of the standard exponential distribution

        c = np.diag(quantile)

    else:
        raise AssertionError('wrong regime of covariance matrices')

    return c

def data_random_split(data, sizes):
    """Generates a random split of a given dataset

    Takes a (x, y) tuple where x and y are numpy arrays
    """
    x, y = data
    # Shuffle data
    idx = np.arange(x.shape[0])
    np.random.shuffle(idx)
    x_shuffled = x[idx]
    # TODO: move this out of here and in to the data generating funtions
    y_shuffled = np.expand_dims(y[idx], 1)

    # Split it up
    split = []
    acc = 0
    for s in sizes:
        subset = (x_shuffled[acc:acc+s], y_shuffled[acc:acc+s])
        split.append(subset)
        acc += s

    return split
