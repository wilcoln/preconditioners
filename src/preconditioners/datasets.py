import torch
import torch.utils.data
from icecream import ic

from preconditioners.utils import generate_centered_linear_gaussian_data, generate_centered_quadratic_gaussian_data

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

    def __init__(self, W_star, w_star, c, n=200, d=600, sigma2=1):
        X, y, _ = generate_centered_quadratic_gaussian_data(W_star, w_star, c, n, d, sigma2)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

def generate_centered_linear_gaussian_data(w_star, c, n=200, d=10, sigma2=1, fix_norm_of_x=False):
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
    xi = np.random.normal(0, sigma2, size=n)

    # generate response
    y = X.dot(w_star) + xi

    return X, y, xi


def generate_centered_quadratic_gaussian_data(W_star, w_star, c, n=200, d=10, sigma2=1):
    """Generates quadratic Gaussian data

    y = X^T W* X + X^T w* + xi
    X: mean zero Gaussian with covariance c
    xi: mean zero Gaussian with std sigma2
    """
    assert W_star.shape == (d, d), 'dimensions error'

    # generate features
    X = np.random.multivariate_normal(mean=np.zeros(d), cov=c, size=n)

    # generate_noise
    xi = np.random.normal(0, sigma2, size=n)

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


def generate_true_parameter(d=10, r2=5, m=None):
    """Generates w_star for the linear and quadratic dataset"""
    if m is None:
        m = np.eye(d)

    assert (m.shape[0] == d) & (m.shape[1] == d)

    w_star = np.random.multivariate_normal(np.zeros(d), r2 / d * m)

    return w_star

def generate_W_star(d=10, r2=5):
    """Generates W_star for the quadratic model

    Return a somewhat random matrix of size (d, d), which does not have degenerate trace or determinant.
    """
    ro = 0.4 + np.random.rand()/2
    c = generate_c(ro=ro, regime='autoregressive', d=d)
    V, D, Vt = np.linalg.svd(c)

    for i in range(len(D)):
        if np.random.rand()<0.25:
            D[i] = D[i]*0.5
        elif np.random.rand()>0.75:
            D[i] = D[i]*2
    D = np.abs(D + np.random.multivariate_normal(np.zeros(d), np.diag(D)/10))
    largest_eval = np.max(D)
    D = D/largest_eval

    W = V.dot(np.diag(D).dot(Vt))*np.sqrt(r2/d) # this way x^T W x ~ sqrt(r2/d), which is the same as x^T w_star
    return W


def generate_c(ro=0.25, regime='id', d=600, strong_feature=1, strong_feature_ratio=1 / 2, weak_feature=1):
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
