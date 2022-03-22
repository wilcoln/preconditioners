"""
Sample code automatically generated on 2022-03-15 12:54:45

by geno from www.geno-project.org

from input

parameters
  matrix B symmetric
  matrix X
variables
  matrix C
min
  tr(inv(X*C*C'*X')*X*C*C'*B*C*C'*X'*inv(X*C*C'*X'))


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
try:
    from genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)



class GenoNLP:
    def __init__(self, B, X, CInit, np):
        self.np = np
        self.B = B
        self.X = X
        self.CInit = CInit
        assert isinstance(B, self.np.ndarray)
        dim = B.shape
        assert len(dim) == 2
        self.B_rows = dim[0]
        self.B_cols = dim[1]
        assert isinstance(X, self.np.ndarray)
        dim = X.shape
        assert len(dim) == 2
        self.X_rows = dim[0]
        self.X_cols = dim[1]
        assert isinstance(CInit, self.np.ndarray)
        dim = CInit.shape
        assert len(dim) == 2
        self.C_rows = dim[0]
        self.C_cols = dim[1]
        self.C_size = self.C_rows * self.C_cols
        # the following dim assertions need to hold for this problem
        assert self.B_rows == self.C_rows == self.B_cols == self.X_cols

    def getLowerBounds(self):
        bounds = []
        bounds += [-inf] * self.C_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [inf] * self.C_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        return self.CInit.reshape(-1)

    def variables(self, _x):
        C = _x
        C = C.reshape(self.C_rows, self.C_cols)
        return C

    def fAndG(self, _x):
        C = self.variables(_x)
        T_0 = self.np.linalg.inv((((self.X).dot(C)).dot(C.T)).dot(self.X.T))
        T_1 = (self.X.T).dot(T_0)
        T_2 = ((((T_1).dot(T_0)).dot(self.X)).dot(C)).dot(C.T)
        T_3 = ((T_2).dot(self.B.T)).dot(C)
        T_4 = (((T_1).dot(self.X)).dot(C)).dot(C.T)
        T_5 = ((T_2).dot(self.B)).dot(C)
        f_ = self.np.trace(((((((((T_0).dot(self.X)).dot(C)).dot(C.T)).dot(self.B)).dot(C)).dot(C.T)).dot(self.X.T)).dot(T_0))
        g_0 = (((((T_3 - ((((((T_3).dot(C.T)).dot(self.X.T)).dot(T_0)).dot(self.X)).dot(C) + ((((((((T_4).dot(self.B)).dot(C)).dot(C.T)).dot(self.X.T)).dot(T_0)).dot(T_0)).dot(self.X)).dot(C))) + (((((((self.B).dot(C)).dot(C.T)).dot(self.X.T)).dot(T_0)).dot(T_0)).dot(self.X)).dot(C)) + (((((((self.B.T).dot(C)).dot(C.T)).dot(self.X.T)).dot(T_0)).dot(T_0)).dot(self.X)).dot(C)) + T_5) - (((((((((T_4).dot(self.B.T)).dot(C)).dot(C.T)).dot(self.X.T)).dot(T_0)).dot(T_0)).dot(self.X)).dot(C) + (((((T_5).dot(C.T)).dot(self.X.T)).dot(T_0)).dot(self.X)).dot(C)))
        g_ = g_0.reshape(-1)
        return f_, g_

def var_solve(B, X, CInit, np, geno_tol = 1e-6):
    start = timer()
    NLP = GenoNLP(B, X, CInit, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    # These are the standard solver options, they can be omitted.
    options = {'eps_pg' : geno_tol,
               'max_iter' : 3000,
               'm' : 10,
               'ls' : 0,
               'verbose' : 10  # Set it to 0 to fully mute it.
              }

    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.1.0')
        result = minimize(NLP.fAndG, x0, lb=lb, ub=ub, options=options, np=np)
    else:
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=list(zip(lb, ub)))

    # assemble solution and map back to original problem
    C = NLP.variables(result.x)
    elapsed = timer() - start
    print('solving took %.3f sec' % elapsed)
    return result, C

def generateRandomData(np):
    np.random.seed(0)
    B = np.random.randn(3, 3)
    B = 0.5 * (B + B.T)  # make it symmetric
    X = np.random.randn(3, 3)
    CInit = np.random.randn(3, 3)
    return B, X, CInit