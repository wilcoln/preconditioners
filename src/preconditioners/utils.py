from mimetypes import init
import warnings

import numpy as np
import pandas as pd
import torch, scipy
from numpy.random import normal
from sklearn.covariance import LedoitWolf, GraphicalLasso
from sklearn.model_selection import train_test_split
from torch import nn
import torch.nn.functional as F
from preconditioners.cov_approx.variance import var_solve

import preconditioners.settings


def sq_loss(y_pred, y):
    return np.linalg.norm(y_pred - y) ** 2


def calculate_risk(w_star, c, w=0):
    return (w - w_star).dot(c.dot(w - w_star))


def calculate_risk_rf(a, w_star, c, cov_z, cov_zx):
    return a.dot(cov_z.dot(a)) + w_star.dot(c.dot(w_star)) - 2 * a.dot(cov_zx.dot(w_star))


def compute_best_achievable_interpolator(X, y, c_inv, m, snr, crossval_param=100):
    """ If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error on for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosed as an estimate of the signal-to-noise ratio."""

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


def compute_best_achievable_interpolator_rf(X, Z, y, cov_z_inv, cov_zx, m, snr, crossval_param):
    """ If snr is passed as a list then for each entry in snr, this function splits
        X, y into a train-test split crossval_param times and calculates the
        average crossvalidated error on for the given entry in snr. The
        average of the first three entries which minimize the crossvalidated error
        is chosed as an estimate of the signal-to-noise ratio."""

    d = X.shape[1]
    n = X.shape[0]
    N = Z.shape[1]

    # calculate the best_achievable interpolator according to formula in paper
    m_1 = cov_z_inv

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


def generate_true_parameter(d=600, r2=5, m=None):
    if m is None:
        m = np.eye(d)

    assert (m.shape[0] == d) & (m.shape[1] == d)

    w_star = np.random.multivariate_normal(np.zeros(d), r2 / d * m)

    return w_star


def generate_c(ro=0.25,
               regime='id',
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


def generate_c_inv_empir(X, empir, alpha=0.25, mu = 0.1, geno_tol = 1e-6, X_extra = None, X_test = None):

    '''
    Generates approximation of the inverse Fisher / covariance matrix. 

    Parameters

    ---------------

    X  : n x p matrix, where row i of X is the gradient of the model at the ith data point.

    empir :  Specifies the method used to approximate the Fisher / covariance.

    alpha :  Regularization parameter in the Graphical Lasso, whenever glasso is used (e.g. 
            if empir = 'gl' or empir = 'variance_gl').

    mu :  Regularization parameter in damping methods. E.g. in empir = 'extra', the result is
            c_inv = X.T.dot(X)/n + mu * np.eye(p) (if X_extra = None)

    geno_tol :  Tolerance for the genosolver which is a framework for optimization problems.

    X_extra : n_extra x p matrix, where row i of X is the gradient of the model at the ith extra 
            data point. Used in methods where the matrix is approximated using extra data.

    X_test : n_test x p matrix, where row i of X is the gradient of the model at the ith test 
            data point. Used in methods where the matrix is approximated using test data.

    '''

    if empir == 'lw':
        lw = LedoitWolf(assume_centered=True).fit(X)
        c_inv_e = lw.precision_

    elif empir == 'gl':
        gl = GraphicalLasso(assume_centered=True, alpha=alpha, tol=1e-4).fit(X)
        c_inv_e = gl.precision_
    
    elif empir == 'variance_gl':
        # compute glasso
        gl = GraphicalLasso(assume_centered=True, alpha=alpha, tol=1e-4).fit(X)
        B = gl.covariance_

        n, d = X.shape
        cov_empir = X.T.dot(X) / n
        C_init = initialize_C(cov_empir = cov_empir, e_1=0.1, e_2=0.5, ro=0.2)
        _, C = var_solve(B = B, X=X, CInit = C_init, np=np, geno_tol=geno_tol)
        c_inv_e = C.dot(C.T)

    elif empir == 'variance_id':
        n, d = X.shape
        B = np.eye(d)        
        cov_empir = X.T.dot(X) / n
        C_init = initialize_C(cov_empir = cov_empir, e_1=0.1, e_2=0.5, ro=0.2)
        _, C = var_solve(B = B, X=X, CInit = C_init, np=np, geno_tol = geno_tol)
        c_inv_e = C.dot(C.T)
    
    elif empir == 'extra':
        n, d = X.shape
        n_extra, d_extra = X_extra.shape
        assert X_extra is not None, 'need to provide extra data'
        assert d_extra == d, 'extra_data needs to have the same dimension as X'
        return np.linalg.inv( (X.T.dot(X) + X_extra.T.dot(X_extra)) / (n + n_extra) + mu*np.eye(d) )
    
    elif empir == 'test':
        n, d = X.shape
        n_test, d_test = X_test.shape
        assert X_test is not None, 'need to provide test data'
        assert d_test == d, 'test_data needs to have the same dimension as X'
        return np.linalg.inv( (X.T.dot(X) + X_test.T.dot(X_test)) / (n + n_test) + mu*np.eye(d) )

    elif empir == 'variance_extra':
        n, d = X.shape
        n_extra, d_extra = X_extra.shape
        assert X_extra is not None, 'need to provide extra data'
        assert d_extra == d, 'extra_data needs to have the same dimension as X'
        cov_empir = X.T.dot(X) / n
        B = ( X.T.dot(X) + X_extra.T.dot(X_extra) ) / (n + n_extra) + mu*np.eye(d)
        C_init = initialize_C(cov_empir = X.T.dot(X)/n, e_1=0.1, e_2=0.5, ro=0.2)
        _, C = var_solve(B = B, X=X, CInit = C_init, np=np, geno_tol=geno_tol)
        c_inv_e = C.dot(C.T)
        
    else:
        raise AssertionError('specify regime of empirical approximation')

    return c_inv_e


def initialize_C(cov_empir, e_1=0.1, e_2=0.5, ro=0.2):
    #TODO: initializing at something which I don't have to invert is probably not the best idea better
    d = cov_empir.shape[0]
    cov_inv = np.linalg.inv(cov_empir + e_1 * np.eye(d))
    C_init = scipy.linalg.cholesky(cov_inv) + e_2 * generate_c(ro=0.2,
                                                                regime='autoregressive',
                                                                d=d,
                                                                )
    return C_init                                                                


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
    param_grads = [param_gradient(y, param) for param in model.parameters()]

    return torch.cat([grad.view(-1) for grad in param_grads]).view(-1, 1)


def param_gradient(y, param, grad_outputs=None):
    """Compute dy/dx @ grad_outputs"""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    return torch.autograd.grad(y, param, grad_outputs=grad_outputs, create_graph=True)[0]


def batch_jacobian(net, x):
    noutputs = net.out_dim
    x = x.unsqueeze(1)  # b, 1 ,in_dim
    n = x.size()[0]
    x = x.repeat(1, noutputs, 1) # b, out_dim, in_dim
    x.requires_grad_(True)
    y = net(x)
    input_val = torch.eye(noutputs).reshape(1, noutputs, noutputs).repeat(n, 1, 1)
    y.backward(input_val)
    return x.grad.data


def model_gradient_2(model, x):
    model.zero_grad()
    batch_jacobian(model, x)
    grads = torch.cat([param.grad.view(-1) for param in model.parameters()]).view(-1, 1).detach()
    model.zero_grad()
    return grads


class SLP(nn.Module):
        """ Single Layer Perceptron for regression. """

        def __init__(self, in_channels, activation=False):
            super().__init__()
            self.activation = activation
            self.layer = nn.Linear(in_channels, 1, bias=False)
            self.out_dim = 1

        def forward(self, x):
            """ Forward pass of the MLP. """
            x = self.layer(x)
            if self.activation:
                return F.relu(x)
            return x

        def reset_parameters(self):
            return self.layer.reset_parameters()


class MLP(nn.Module):
    """ Single Layer Perceptron for regression. """

    def __init__(self, in_channels, num_layers=2, hidden_layer_size=100):
        super().__init__()
        self.in_layer = nn.Linear(in_channels, hidden_layer_size, bias=False)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_layer_size, hidden_layer_size)
            for _ in range(num_layers - 2)
        ])
        self.output_layer = nn.Linear(hidden_layer_size, 1, bias=False)
        self.out_dim = 1

    def forward(self, x):
        """ Forward pass of the MLP. """
        x = self.in_layer(x)
        x = F.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
        return x

    def reset_parameters(self):
        self.in_layer.reset_parameters()
        for layer in self.hidden_layers:
            layer.reset_parameters()
        self.output_layer.reset_parameters()




