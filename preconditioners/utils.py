import numpy as np
import pandas as pd
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
