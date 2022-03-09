import sys

import numpy as np
import scipy

from preconditioners.utils import generate_c, generate_centered_gaussian_data


def log_lik(cov_inv, cov_empir):
    # TODO: nimplement cholesky decomposition functionality for speedup of computing determinant
    return np.log(np.linalg.det(cov_inv) + 1e-10) - np.trace(cov_empir.dot(cov_inv))


def regul(cov_inv, X):
    # use the 1/||(I - X^T(XX^T)^-1X)cov_invX^T ||_{Tr}^2 regularization because 
    # this function is convex in cov_inv
    temp_1 = (np.eye(X.shape[1]) - X.T.dot(np.linalg.inv(X.dot(X.T))).dot(X))
    temp_2 = temp_1.dot(cov_inv.dot(X.T))
    return 1 / np.trace(temp_2.dot(temp_2.T))


def loss(cov_inv, X, cov_empir, regul_lambda):
    # subtracting the regularization term from the log-likelihood because we want to maximize the log-likelihood
    return log_lik(cov_inv, cov_empir) - regul_lambda * regul(cov_inv, X)


def grad_log_lik(C, cov_empir):
    # using the parametrization cov_inv = CC^T and computing gradient with respect to C
    return 2 * (np.linalg.inv(C.dot(C.T)) - cov_empir).dot(C)


def grad_regul(C, X, cov_empir):
    # using the parametrization cov_inv = CC^T and computing gradient with respect to C
    # TODO: iI think it makes a computational difference if I do A.dot(B.dot(C)) or A.dot(B).dot(C). Check this and possibly optimize.

    XtX = cov_empir * X.shape[0]
    CCt = C.dot(C.T)
    A = np.eye(X.shape[1]) - X.T.dot(np.linalg.inv(X.dot(X.T)).dot(X))
    temp_1 = A.dot(A.dot(CCt.dot(XtX)))
    scalar = 2 / np.trace(temp_1.dot(CCt)) ** 2
    return -scalar * (temp_1 + temp_1.T).dot(C)


def grad_loss(C, cov_empir, regul_lambda, X):
    # using the parametrization cov_inv = CC^T and computing gradient with respect to C
    assert np.linalg.cond(C) < 1 / sys.float_info.epsilon, 'C should be invertible'
    assert (cov_empir == cov_empir.T).all(), 'cov_empir should be symmetric'

    # note there is a minus also inside grad_regul.
    return grad_log_lik(C, cov_empir) - regul_lambda * grad_regul(C, X, cov_empir)
