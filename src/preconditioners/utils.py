import warnings

import numpy as np
import pandas as pd
import torch
from numpy.random import normal
from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.model_selection import train_test_split


def sq_loss(y_pred, y):
    return np.linalg.norm(y_pred - y) ** 2


def calculate_risk(w_star, c, w=0):
    return (w - w_star).dot(c.dot(w - w_star))


def calculate_risk_rf(a, w_star, c, cov_z, cov_zx):
    return a.dot(cov_z.dot(a)) + w_star.dot(c.dot(w_star)) - 2 * a.dot(cov_zx.dot(w_star))


def compute_best_achievable_interpolator(X, y, c, m, snr, crossval_param=100):
    """ If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error on for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosed as an estimate of the signal-to-noise ratio."""

    c_inv = np.linalg.inv(c)
    d = X.shape[1]
    n = X.shape[0]

    if type(snr) == np.ndarray or type(snr) == list:

        # initialize dataframe where we save results
        df = pd.DataFrame([], columns=['mu', 'error'])

        for mu in snr:

            error_crossvalidated = 0

            for j in range(crossval_param):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)  # random train test split

                n_train = X_train.shape[0]
                n_test = X_test.shape[0]

                # calculate the best_achievable interpolator according to formula in paper
                auxi_matrix_train = np.linalg.inv(np.eye(n_train) + (mu / d) * X_train.dot(m.dot(X_train.T)))
                auxi_matrix_train_2 = ((mu / d) * m.dot(X_train.T) + (c_inv.dot(X_train.T)).dot(
                    np.linalg.inv(X_train.dot(c_inv.dot(X_train.T)))))
                w_e_train = auxi_matrix_train_2.dot(auxi_matrix_train.dot(y_train))

                y_test_pred = X_test.dot(w_e_train)

                error_crossvalidated += (np.linalg.norm(y_test - y_test_pred) ** 2) / n_test

            error_crossvalidated = error_crossvalidated / crossval_param

            df = df.append(pd.DataFrame(np.array([[mu, error_crossvalidated]]), columns=['mu', 'error']))

        df = df.sort_values('error', ascending=True)

        snr = np.mean(df['mu'].iloc[:3].values)

    # calculate the best_achievable interpolator according to formula in paper
    auxi_matrix = np.linalg.inv(np.eye(n) + (snr / d) * X.dot(m.dot(X.T)))
    auxi_matrix_2 = ((snr / d) * m.dot(X.T) + (c_inv.dot(X.T)).dot(np.linalg.inv(X.dot(c_inv.dot(X.T)))))
    w_e = auxi_matrix_2.dot(auxi_matrix.dot(y))

    return w_e


def compute_best_achievable_interpolator_rf(X, Z, y, cov_z, cov_zx, m, snr, crossval_param):
    """ If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error on for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosed as an estimate of the signal-to-noise ratio."""

    d = X.shape[1]
    n = X.shape[0]
    N = Z.shape[1]

    # calculate the best_achievable interpolator according to formula in paper
    m_1 = np.linalg.inv(cov_z)

    m_21 = cov_zx.dot(m.dot(X.T))
    m_22 = Z.T.dot(np.linalg.inv(Z.dot(m_1.dot(Z.T))))
    m_23 = (d / snr) * np.eye(n) + X.dot(m.dot(X.T)) - Z.dot(m_1.dot(cov_zx.dot(m.dot(X.T))))

    m_2 = m_21 + m_22.dot(m_23)

    m_3 = np.linalg.inv((d / snr) * np.eye(n) + X.dot(m.dot(X.T)))

    w = m_1.dot(m_2.dot(m_3.dot(y)))

    # print(np.linalg.norm(w - np.linalg.lstsq(Z,y,rcond = None)[0]))

    return w


def compute_best_achievable_estimator_rf(X, y, cov_z, cov_zx, m, snr, crossval_param):
    d = X.shape[1]
    n = X.shape[0]

    m_1 = np.linalg.inv(cov_z).dot(cov_zx.dot(m.dot(X.T)))
    # why d/snr and not snr/d
    m_2 = np.linalg.inv((d / snr) * np.eye(n) + X.dot(m.dot(X.T)))

    return m_1.dot(m_2.dot(y))


def compute_optimal_ridge_regressor(X, y, snr):
    d = X.shape[1]
    n = X.shape[0]

    m_1 = X.T
    m_2 = np.linalg.inv((d / snr) * np.eye(n) + X.dot(X.T))

    return m_1.dot(m_2.dot(y))


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

    X = np.random.multivariate_normal(mean=np.zeros(d), cov=c, size=n)

    if fix_norm_of_x:
        X = X * np.sqrt(d) / np.linalg.norm(X, axis=1)[:, None]

    # print warning if X is not on the sphere
    if any(abs(np.linalg.norm(X, axis=1) - np.sqrt(d)) > 1e-5):
        warnings.warn('Warning, norms of datapoints are not sqrt(d)')

    # generate_noise
    xi = np.random.multivariate_normal(np.zeros(n), sigma2 * np.eye(n))

    # generate response
    y = X.dot(w_star) + xi

    return X, y, xi


def flatten_params(parameters):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**

    """
    l = [torch.flatten(p) for p in parameters]
    indices = []
    s = 0
    for p in l:
        size = p.shape[0]
        indices.append((s, s + size))
        s += size
    flat = torch.cat(l).view(-1, 1)
    return {"params": flat, "indices": indices}


def recover_flattened(flat_params, indices, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(model.parameters()):
        l[i] = l[i].view(*p.shape)
    return l


def model_gradient(y, model):
    param_grads = []

    for param in model.parameters():
        param_grads.append(param_gradient(y, param))

    return torch.cat([grad.view(-1) for grad in param_grads]).view(-1, 1)


def param_gradient(y, param, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    return torch.autograd.grad(y, param, grad_outputs=grad_outputs, create_graph=True)[0]