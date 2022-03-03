import unittest
import numpy as np
import scipy
from preconditioners.impl_cov_approx import *
from preconditioners.datagen import generate_c, generate_centered_gaussian_data

class TestImplCov(unittest.TestCase):

    longMessage = True

    def __init__(self, *args, **kwargs):
        super(TestImplCov, self).__init__(*args, **kwargs)
        self.n = 50
        self.d = 150
        self.c = generate_c(ro=0.5,
                    regime='autoregressive',
                    n=self.n,
                    d=self.d
                    )
        self.w_star = np.random.multivariate_normal(mean = np.zeros(self.d),cov = np.eye(self.d))    
        self.X, self.y, self.xi = generate_centered_gaussian_data(self.w_star,
                        self.c,
                        n = self.n,
                        d = self.d,
                        sigma2=1,
                        fix_norm_of_x = False)

    def test_log_lik(self):
        cov_inv = np.eye(self.d)
        cov_empir = np.eye(self.d)
        self.assertAlmostEqual(log_lik(cov_inv, cov_empir), np.log(np.linalg.det(cov_inv)) - self.d)

    def test_log_lik_2(self):
        cov_inv = self.c
        for i in range(self.d):
            for j in range(self.d):
                if i == j:
                    cov_inv[i,j] = 2
                if i<j:
                    cov_inv[i,j] = 0
        cov_empir = np.linalg.inv(cov_inv)
        self.assertAlmostEqual(log_lik(cov_inv, cov_empir), self.d*np.log(2) - self.d)

    def test_regul_1(self):
        cov_inv = np.eye(self.d)
        try:
            reg = regul(cov_inv, self.X)
        except Exception as e:
            self.assertIsInstance(e, ZeroDivisionError)
        if reg is not None:
            self.assertTrue(10**10 < np.linalg.norm(reg) or np.isnan(reg).all())
    
    def test_regul_2(self):
        cov_inv = np.linalg.inv(self.c)
        try:
            reg = regul(cov_inv, np.eye(self.d))
        except Exception as e:
            self.assertIsInstance(e, ZeroDivisionError)
        if reg is not None:
            self.assertTrue(10**10 < np.linalg.norm(reg) or np.isnan(reg).all())

    def test_regul_3(self):
        cov_inv = np.linalg.inv(self.X.T.dot(self.X)/self.n + np.eye(self.d))
        try:
            reg = regul(cov_inv, self.X)
        except Exception as e:
            self.assertIsInstance(e, ZeroDivisionError)
        if reg is not None:
            self.assertTrue(10**10 < np.linalg.norm(reg) or np.isnan(reg).all())
    
    def test_regul_4(self):
        cov_inv = np.linalg.inv(self.c)
        self.assertTrue(regul(cov_inv, self.X) > 0)

    def test_grad_lok_lik(self):
        C = np.eye(self.d)
        cov_empir = self.X.T.dot(self.X)/self.n
        np.testing.assert_array_equal(grad_log_lik(C, cov_empir), 2*(np.eye(self.d) - cov_empir))

    def test_grad_regul(self):
        C = self.c
        X = np.eye(self.d)
        cov_empir = self.X.T.dot(self.X)/self.n
        try:
            grad_reg = grad_regul(C, X, cov_empir)
        except Exception as e:
            self.assertIsInstance(e, ZeroDivisionError)
        if grad_reg is not None:
            self.assertTrue(10**10 < np.linalg.norm(grad_reg) or np.isnan(grad_reg).all())
        
    
    def test_grad_regul_2(self):
        C = np.eye(self.d)
        cov_empir = self.X.T.dot(self.X)/self.n
        try:
            grad_reg = grad_regul(C, self.X, cov_empir)
        except Exception as e:
            self.assertIsInstance(e, ZeroDivisionError)
        if grad_reg is not None:
            self.assertTrue(10**10 < np.linalg.norm(grad_reg) or np.isnan(grad_reg).all())

    def test_grad_regul_3(self):
        cov_empir = self.X.T.dot(self.X)/self.n
        cov_inv = np.linalg.inv(cov_empir + np.eye(self.d))
        C = scipy.linalg.sqrtm(cov_inv)
        try:
            grad_reg = grad_regul(C, self.X, cov_empir)
        except Exception as e:
            self.assertIsInstance(e, ZeroDivisionError)
        if grad_reg is not None:
            self.assertTrue(10**10 < np.linalg.norm(grad_reg) or np.isnan(grad_reg).all())

    def test_grad_regul_4(self):
        cov_inv = np.linalg.inv(self.c)
        C = scipy.linalg.sqrtm(cov_inv)
        cov_empir = self.X.T.dot(self.X)/self.n
        grad_regul(C, self.X, cov_empir)

    def test_grad_regul_5(self):
        cov_empir = self.X.T.dot(self.X)/self.n
        cov_inv = np.linalg.inv(cov_empir + 0.1*np.eye(self.d))
        C = scipy.linalg.sqrtm(cov_inv) + 0.1*generate_c(ro=0.1,
                    regime='autoregressive',
                    n=self.n,
                    d=self.d,
                    )
        grad_regul(C, self.X, cov_empir)

    def test_grad_loss(self):
        cov_inv = np.linalg.inv(self.c)
        C = scipy.linalg.sqrtm(cov_inv)
        cov_empir = self.X.T.dot(self.X)/self.n
        regul_lambda = 0
        np.testing.assert_array_almost_equal(
            grad_loss(C, cov_empir, regul_lambda, self.X),
            grad_log_lik(C, cov_empir)  
        )
    
    def test_grad_loss_2(self):
        cov_inv = np.linalg.inv(self.c)
        C = scipy.linalg.sqrtm(cov_inv)
        cov_empir = self.X.T.dot(self.X)/self.n
        regul_lambda = 12
        np.testing.assert_array_almost_equal(
            grad_loss(C, cov_empir, regul_lambda, self.X) - grad_log_lik(C,cov_empir),
            -12*grad_regul(C, self.X, cov_empir)
        )

    def test_grad_loss_3(self):
        cov_inv = np.linalg.inv(self.c)
        C = np.linalg.cholesky(cov_inv)
        cov_empir = self.X.T.dot(self.X)/self.n
        regul_lambda = 12
        np.testing.assert_array_almost_equal(
            grad_loss(C, cov_empir, regul_lambda, self.X) - grad_log_lik(C,cov_empir),
            -12*grad_regul(C, self.X, cov_empir)
        )

