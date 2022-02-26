
import warnings

import numpy as np
from numpy.random import normal
from sklearn.covariance import LedoitWolf, GraphicalLasso


def generate_m(c, source_condition='id'):
    if source_condition == 'id':
        m = np.eye(c.shape[0])

    elif source_condition == 'easy':
        m = c

    elif source_condition == 'hard':
        m = np.linalg.inv(c)

    return m


def generate_true_parameter(n=200, d=600, r2=5, m=None):
    if m is None:
        m = np.eye(d)

    assert (m.shape[0] == d) & (m.shape[1] == d)

    w_star = np.random.multivariate_normal(np.zeros(d), r2 / d * m)

    return w_star


def generate_c(ro=0.25,
               regime='id',
               n=200,
               d=600,
               strong_feature=1,
               strong_feature_ratio=1 / 2,
               weak_feature=1):
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


def generate_c_empir(X, empir, alpha=0.25):
    if empir == 'basic':
        c_e = X.transpose().dot(X) / len(X)

    elif empir == 'lw':

        lw = LedoitWolf(assume_centered=True).fit(X)
        c_e = lw.covariance_

    elif empir == 'gl':
        gl = GraphicalLasso(assume_centered=True, alpha=alpha, tol=1e-4).fit(X)
        c_e = gl.covariance_

    else:
        raise AssertionError('specify regime of empirical approximation')

    return c_e


def generate_centered_gaussian_data(w_star, c, n=200, d=600, sigma2=1, fix_norm_of_x=False):
    assert len(w_star) == d, 'dimensions error'

    # generate features
    X = np.zeros((n, d))

    for i in range(n):
        X[i] = np.random.multivariate_normal(np.zeros(d), c)

        if fix_norm_of_x:
            X[i] = X[i] * np.sqrt(d) / np.linalg.norm(X[i])

    # print warning if X is not on the sphere
    if any(abs(np.linalg.norm(X, axis=1) - np.sqrt(d)) > 1e-5):
        warnings.warn('Warning, norms of datapoints are not sqrt(d)')

    # generate_noise
    xi = np.random.multivariate_normal(np.zeros(n), sigma2 * np.eye(n))

    # generate response
    y = X.dot(w_star) + xi

    return X, y, xi
