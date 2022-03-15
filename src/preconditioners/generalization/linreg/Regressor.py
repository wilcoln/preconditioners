import numpy as np
import scipy.linalg as sc
import warnings


class LinearRegressor:


    def __init__(self, init = None):

        self.fitted = False
        self.d = None

        if init is not None:
            self.initialized = True

        # initialize parameter
        self.w = init



    def fit(self, X, y, matrix = None):

        self.fitted = True
        self.d = X.shape[1]

        if matrix is None:

            matrix = np.eye(self.d)

        assert matrix.shape[0] == self.d, 'wrong dimension of matrix'

        matrix_half = sc.sqrtm(matrix) # compute square root of matrix
        matrix_mhalf = np.linalg.inv(matrix_half) # compute its inverse

        assert len(y.shape) == 1, 'output needs to be one-dimensional'
        assert X.shape[0] == len(y), 'dimension of X and y dont match'

        # We directly compute the limit using the implicit bias of mirror (and gradient) descent
        # if matrix is the identity, this is just the minimum-norm solution
        Z = X.dot(matrix_mhalf)
        self.w = matrix_mhalf.dot(np.linalg.lstsq(Z, y, rcond=None)[0])




    def predict(self, X):

        assert ((self.fitted) | (self.initialized)), 'need to train or initlaize the model first'

        assert X.shape[1] == len(self.w), 'data must have the same dimension as the training data (or initialization)'

        return X.dot(self.w)




class RandomFeaturesRegressor:

    def __init__(self, nonlinearity = 'ReLU',  init_N = 200, init_theta = None, init_w = None, init_d = None, fix_norm_of_theta = True):

        self.fitted = False

        self.w = init_w # supposed to be of dimension N
        self.theta = init_theta # supposed to be Nxd matrix where d is dimension of data
        self.fix_norm_of_theta = fix_norm_of_theta

        if self.theta is not None:
            self.N = init_theta.shape[0]
            self.d = init_theta.shape[1]

            # print warning if theta is not on the sphere
            if any( abs(np.linalg.norm(self.theta, axis = 1) - np.sqrt(self.d)) > 1e-5):
                warnings.warn('Warning, norms of theta are not sqrt(d)')
        else:
            self.N = init_N
            self.d = init_d

        # initialize nonlinearity
        if nonlinearity == 'ReLU':
            self.nonlinearity = lambda x: np.maximum(x, 0)
        else:
            self.nonlinearity = lambda x: x

        # assert equal dimeension
        if ( (self.theta is not None) & (self.w is not None) ):
            assert self.theta.shape[0] == len(self.w)
            self.initialized = True
        else:
            self.initialized = False



    def initialize_theta(self, init_d = None):

        # initialize d
        if self.d is None:
            if init_d is None:
                raise AssertionError('need to initialize d')
            self.d = init_d

        # initialize_theta
        self.theta = np.zeros((self.N,self.d))

        for i in range(self.N):
            self.theta[i] = np.random.multivariate_normal(np.zeros(self.d),np.eye(self.d))

            if self.fix_norm_of_theta==True:
                self.theta[i] = self.theta[i]*np.sqrt(self.d)/np.linalg.norm(self.theta[i])



    def return_Z(self, X):

        assert self.theta is not None, 'theta must be initialized'
        assert X.shape[1] == self.theta.shape[1], 'data must have the same dimension as the first layer'

        Z = self.nonlinearity( X.dot(self.theta.T)/np.sqrt(self.d) ) # shape nxN
        return Z



    def fit(self, X, y):

        self.fitted = True
        self.n = X.shape[0]
        self.d = X.shape[1]

        # if we haven't initialized theta already then do so now
        if self.theta is None:
            raise AssertionError('need to initialize theta')

        assert len(y.shape) == 1
        assert X.shape[0] == len(y)
        assert X.shape[1] == self.d

        Z = self.return_Z(X)
        self.w = np.linalg.lstsq(Z, y, rcond=None)[0]



    def predict(self, X):

        assert ((self.fitted) | (self.initialized)), 'need to train or initlaize the model first'
        assert X.shape[1] == self.theta.shape[1], 'data must have the same dimension as the first layer'

        # first layer, shape Nxn
        l1 = self.nonlinearity(self.theta.dot(X.T)/np.sqrt(self.d))

        return self.w.dot(l1) #shape n
