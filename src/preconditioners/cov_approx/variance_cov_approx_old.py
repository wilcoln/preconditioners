from __future__ import division, print_function, absolute_import

import numpy as np

# P = C.dot(C.T)
# X = X
# B = Sigma (cov_empir)

def fAndG(B, C, X, a):
    # this one uses the log barrier
    assert isinstance(B, np.ndarray)
    dim = B.shape
    assert len(dim) == 2
    B_rows = dim[0]
    B_cols = dim[1]
    assert isinstance(C, np.ndarray)
    dim = C.shape
    assert len(dim) == 2
    C_rows = dim[0]
    C_cols = dim[1]
    assert isinstance(X, np.ndarray)
    dim = X.shape
    assert len(dim) == 2
    X_rows = dim[0]
    X_cols = dim[1]
    if isinstance(a, np.ndarray):
        dim = a.shape
        assert dim == (1, )
    assert B_cols == C_rows == X_cols == B_rows

    T_0 = np.linalg.inv((((X).dot(C)).dot(C.T)).dot(X.T))
    T_1 = (X.T).dot(T_0)
    T_2 = ((((((T_1).dot(T_0)).dot(X)).dot(C)).dot(C.T)).dot(B)).dot(C)
    T_3 = (((((((((((T_1).dot(X)).dot(C)).dot(C.T)).dot(B)).dot(C)).dot(C.T)).dot(X.T)).dot(T_0)).dot(T_0)).dot(X)).dot(C)
    T_4 = (((((T_2).dot(C.T)).dot(X.T)).dot(T_0)).dot(X)).dot(C)
    T_5 = (C).dot(C.T)
    t_6 = np.linalg.det(T_5)
    t_7 = (1 / (10 ** 10))
    functionValue = (np.trace(((((((((T_0).dot(X)).dot(C)).dot(C.T)).dot(B)).dot(C)).dot(C.T)).dot(X.T)).dot(T_0)) - (a * np.log((t_7 + t_6))))
    gradient = (((((2 * T_2) - (T_4 + T_3)) + (2 * (((((((B).dot(C)).dot(C.T)).dot(X.T)).dot(T_0)).dot(T_0)).dot(X)).dot(C))) - (T_3 + T_4)) - (((2 * a) / (t_6 + t_7)) * ((np.linalg.det(T_5) * np.linalg.inv(T_5))).dot(C)))

    return functionValue, gradient

def invbarrier(C):
    assert isinstance(C, np.ndarray)
    dim = C.shape
    assert len(dim) == 2
    C_rows = dim[0]
    C_cols = dim[1]

    T_0 = (C).dot(C.T)
    t_1 = np.linalg.det(T_0)
    functionValue = (1 / t_1)
    #gradient = -((2 / (t_1 ** 2)) * ((np.linalg.det(T_0) * np.linalg.inv(T_0))).dot(C))
    gradient = -((2 / t_1) * (np.linalg.inv(T_0)).dot(C))
    return functionValue, gradient

def fAndG_invbarrier(B, C, X, a):
    # this one uses the 1/x barrier
    assert isinstance(B, np.ndarray)
    dim = B.shape
    assert len(dim) == 2
    B_rows = dim[0]
    B_cols = dim[1]
    assert isinstance(C, np.ndarray)
    dim = C.shape
    assert len(dim) == 2
    C_rows = dim[0]
    C_cols = dim[1]
    assert isinstance(X, np.ndarray)
    dim = X.shape
    assert len(dim) == 2
    X_rows = dim[0]
    X_cols = dim[1]
    if isinstance(a, np.ndarray):
        dim = a.shape
        assert dim == (1, )
    assert B_cols == C_rows == X_cols == B_rows

    T_0 = np.linalg.inv((((X).dot(C)).dot(C.T)).dot(X.T))
    T_1 = (X.T).dot(T_0)
    T_2 = ((((((T_1).dot(T_0)).dot(X)).dot(C)).dot(C.T)).dot(B)).dot(C)
    T_3 = (((((((((((T_1).dot(X)).dot(C)).dot(C.T)).dot(B)).dot(C)).dot(C.T)).dot(X.T)).dot(T_0)).dot(T_0)).dot(X)).dot(C)
    T_4 = (((((T_2).dot(C.T)).dot(X.T)).dot(T_0)).dot(X)).dot(C)

    invbarrier_val, invbarrier_grad = invbarrier(C)
    functionValue = (np.trace(((((((((T_0).dot(X)).dot(C)).dot(C.T)).dot(B)).dot(C)).dot(C.T)).dot(X.T)).dot(T_0)) + a * invbarrier_val)
    gradient = (((((2 * T_2) - (T_4 + T_3)) + (2 * (((((((B).dot(C)).dot(C.T)).dot(X.T)).dot(T_0)).dot(T_0)).dot(X)).dot(C))) - (T_3 + T_4)) + a * invbarrier_grad)

    return functionValue, gradient

def checkGradient(B, C, X, a):
    # numerical gradient checking
    # f(x + t * delta) - f(x - t * delta) / (2t)
    # should be roughly equal to inner product <g, delta>
    t = 1E-6
    delta = np.random.randn(3, 3)
    f1, _ = fAndG(B, C + t * delta, X, a)
    f2, _ = fAndG(B, C - t * delta, X, a)
    f, g = fAndG(B, C, X, a)
    print('approximation error',
          np.linalg.norm((f1 - f2) / (2*t) - np.tensordot(g, delta, axes=2)))

def generateRandomData():
    B = np.random.randn(3, 3)
    B = 0.5 * (B + B.T)  # make it symmetric
    C = np.random.randn(3, 3)
    X = np.random.randn(3, 3)
    a = np.random.randn(1)

    return B, C, X, a

if __name__ == '__main__':
    B, C, X, a = generateRandomData()
    functionValue, gradient = fAndG(B, C, X, a)
    print('functionValue = ', functionValue)
    print('gradient = ', gradient)

    print('numerical gradient checking ...')
    checkGradient(B, C, X, a)
