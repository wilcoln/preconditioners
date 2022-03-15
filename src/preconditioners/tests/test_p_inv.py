import unittest

from preconditioners.cov_approx.impl_cov_approx import *
from preconditioners.utils import generate_c, generate_centered_gaussian_data
from preconditioners.optimizers import PrecondGD

class TestPinv(unittest.TestCase):

    def test_p_inv(self):

        n = 10
        extra_n = 1000
        d = 30
        simga_2 = 1
        ro = 0.5
        regime = 'autoregressive'

        c = generate_c(ro=ro,
                regime=regime,
                n=n,
                d=d
                )

        w_star = np.random.multivariate_normal(mean=np.zeros(d), cov=np.eye(d))
        X, y, xi = generate_centered_gaussian_data(w_star,
                                                    c,
                                                    n=extra_n+n,
                                                    d=d,
                                                    sigma2=simga_2,
                                                    fix_norm_of_x=False)

        X_extra = X[:extra_n, :]
        y_extra = y[:extra_n]
        X = X[extra_n:, :]          
        y = y[extra_n:]
        
        train_size = int(0.7 * len(dataset))
        test_size = int(0.2 * len(dataset))
        extra_size = len(dataset) - train_size - test_size

        # please fill this in 
        mlp = # make it just a linear model with one layer (so it's gradient should be just the features X)
        labeled_data = # X, y
        unlabeled_data = # X_extra, y_extra
        lr = #       
        optimizer = PrecondGD(mlp = mlp, lr=1, labeled_data=labeled_data, unlabeled_data=unlabeled_data)

        p_inv = optimizer._compute_p_inv()
        
        self.assertTrue(
            np.linalg.norm(p_inv - np.linalg.inv(c)) < 0.01,
            msg=f'The error is {np.linalg.norm(p_inv - np.linalg.inv(c))} and \
            the first entries of p_inv are {p_inv[:4,:4]}\
            while the first entries of the true matrix are {np.linalg.inv(c)[:4,:4]}'
        )                 