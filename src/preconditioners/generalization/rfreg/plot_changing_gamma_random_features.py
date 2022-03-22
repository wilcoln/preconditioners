from fileinput import filename
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc
import os, json

from preconditioners.utils import *
from preconditioners.paths import plots_dir
from preconditioners.generalization.linreg.Regressor import LinearRegressor, RandomFeaturesRegressor
from datetime import datetime as dt



def display_risks_gamma_rf(n = 200,
                        n_test = 100,
                        n_extra = 500,
                        n_benchmark = 500,
                        r2=1,
                        sigma2=1,

                        start_gamma = 1.4,
                        end_gamma = 4,
                        gamma_2 = 3, # n/d

                        regime='id',
                        ro=0.5,

                        strong_feature_ratio = 1/2,
                        strong_feature = 1,
                        weak_feature = 1/5,

                        source_condition = 'id',

                        fix_norm_of_theta = True,
                        fix_norm_of_x = True,

                        empir = 'gl',
                        alpha = 0.01,
                        mu = 0.1,
                        geno_tol = 1e-6,

                        include_best_achievable = True,
                        include_best_achievable_empirical_new = True,

                        snr_estimation = list(np.linspace(0.1,1,20))+list(np.linspace(1,10,20)),
                        crossval_param = 100,

                        savefile = False,
                        ):

    ''' This function generates a plot of risk versus gamma (level of overparametrization d/n) for given regime
        of covariance matrices.

        Parameters:

    ------------------------------

    n : int

        Number of datapoints in simulation.

    r2 : int > 0

        Signal. When the prior of the true parameter is isotropic, r^2 is its expected squared norm.

    sigma2 : int > 0

        Variance of the noise variables.

    start_gamma : float > 0

        Smallest value of gamma (N/d) in the plot.

    end_gamma : float > 0

        Largest value of gamma (N/d) in the plot.

    gamma_2 : float > 0

        The ratio n/d.

    regime : 'id', 'autoregressive', 'strong_weak_features', 'exponential'

        Specifies the regime of covariance matrices of the features to be used.

    ro : float in (0,1)

        Parameter used by the 'autoregressive' regime of covariance matrices.

    strong_feature_ratio : float in (0,1)

        Parameter used by the 'strong_weak_features' regime of covariance matrices.
        int(gamma*n*strong_feature_ratio) is the number of 'strong_feature' eigenvalues on the diagonal of the covariance matrix.

    strong_feature: float > 0

        Parameter used by the 'strong_weak_features' regime of covariance matrices.

    weak_feature: float > 0

        Parameter used by the 'strong_weak_features' regime of covariance matrices.

    source_condition : {'id', 'eaesy', 'hard'}

        If 'id' then the covariance matrix of the prior, m, is the idenity. If 'easy' then m = c, if 'hard' then m = c^{-1}
        where c is the covariance matrix of the data.

    fix_norm_of_theta : Boolean

        If true then rows of theta (first layer weights) all have norm sqrt(d).

    empir : {'gl', 'basic', 'lw'}

        Specifies what kind of approximation of the population covariance matrix we use. 'gl' is the GraphicalLasso
        approximation, 'basic' is the standard X^TX/n, 'lw' is the LedoitWolf approximation.

    alpha : float > 0

        Only used if empir = 'gl'. Specifies the regularization of the GraphicalLasso approximation.
        This can be also crossvalidated, using GraphicalLassoCV from sklearn.

    include_best_achievable : Boolean

        If true then includes the best linearly achievable interpolator in the plots.

    include_best_achievable_empirical : Boolean

        If true then includes the empirical approximation of the best linearly achievable interpolator in the plots.

    snr_estimation : list

        The list of possible signal-to-noise ratio that the crossvalidation should try when approximating the snr.
        Only used if best_achievable_empirical = True.

    crossval_param : int

        Number of crossvalidation splits to use when estimating the signal-to-noise ratio. Only used if
        best_achievable_empirical = True.

    savefile : Boolean

        If true then saves the generated plot.

        '''
    saved_args = locals()
    snr = r2/sigma2
    d = int(n/gamma_2)

    # generate sequence of gammas for plotting
    gammas = np.concatenate((np.linspace(start_gamma,start_gamma+(end_gamma-start_gamma)/3,8),np.linspace(start_gamma+(end_gamma-start_gamma)/3+(end_gamma-start_gamma)/15,end_gamma,7)))

    risks = pd.DataFrame([],columns = ['gd', 'achie', 'achie_e'])

    # generate covariance matrix of data
    c = generate_c(ro = ro, regime = regime, d = d,
                    strong_feature = strong_feature,
                    strong_feature_ratio = strong_feature_ratio,
                    weak_feature = weak_feature)

    # generate covariance matrix of the prior of the true parameter
    m = generate_m(c, source_condition = source_condition)

    # generate true parameter
    w_star = generate_true_parameter(d, r2, m = m)

    # generate data from a normal distribution
    X, y, xi = generate_centered_gaussian_data(w_star, c, n, d, sigma2, fix_norm_of_x)
    print('data generated')

    # generate testing data to compute test error on
    X_test, y_test, xi_test = generate_centered_gaussian_data(w_star, c, n_test, d, sigma2, fix_norm_of_x)
    print('extra data generated')

    # generate benchmark data to have as a reference of performance (has to be different than testing data)
    X_benchmark, y_benchmark, xi_benchmark = generate_centered_gaussian_data(w_star, c, n_benchmark, d, sigma2, fix_norm_of_x)
    print('extra data generated')

     # generate extra data to estimate covariance matrix better
    X_extra, y_extra, xi_extra = generate_centered_gaussian_data(w_star, c, n_extra, d, sigma2, fix_norm_of_x)
    print('extra data generated')

    count = 0
    # do experiment for each gamma
    for i in range(len(gammas)):
        count = count+1
        print(f'{count}/15')

        gamma = gammas[i]
        N = int(gamma*d)

        # initialize models
        reg_2 = RandomFeaturesRegressor(init_N = N, init_d = d, fix_norm_of_theta = fix_norm_of_theta)
        reg_2.initialize_theta()
        reg_2.fit(X, y)

        theta = reg_2.theta
        Z = reg_2.return_Z(X)

        if include_best_achievable:
            # estimate cov_z, cov_zx from outside data
            Z_benchmark = reg_2.return_Z(X_benchmark)
            cov_z_inv = np.linalg.inv(Z_benchmark.T.dot(Z_benchmark)/n_benchmark)
            print('benchmark data covariance matrix inverted')
            cov_zx = Z_benchmark.T.dot(X_benchmark)/n_benchmark
            w_o = compute_best_achievable_interpolator_rf(X=X, Z=Z, y=y, cov_z_inv=cov_z_inv, cov_zx=cov_zx, m=m, snr=snr_estimation, crossval_param=crossval_param)
            reg_o = RandomFeaturesRegressor(init_N = N, init_d = d, init_theta = theta, fix_norm_of_theta = fix_norm_of_theta, init_w = w_o)

        if include_best_achievable_empirical_new:
            #TODO: right now we still estimate cov_zx from extra data (but not cov_z). Change this in the future.
            Z_extra = reg_2.return_Z(X_extra)
            cov_zx = Z_extra.T.dot(X_extra)/n_extra
            cov_z_inv_e = generate_c_inv_empir(Z, empir, alpha, mu, geno_tol, X_extra = Z_extra)
            print('covariance matrix estimated')
            w_oe = compute_best_achievable_interpolator_rf(X=X, Z=Z, y=y, cov_z_inv=cov_z_inv_e, cov_zx=cov_zx, m=m, snr=snr_estimation, crossval_param=crossval_param)
            reg_oe = RandomFeaturesRegressor(init_N = N, init_d = d, init_theta = theta, fix_norm_of_theta = fix_norm_of_theta, init_w = w_oe)

        # It is important to have completely unseen data here, and not the same extra data
        # which is used to estimate the covariance matrix. If the test data is used to estimate
        # the covariance matrix, then the test error is smaller!.
        # Could this actually be used to a benefit?
        risk_2 = sq_loss(reg_2.predict(X_test), y_test)/len(y_test) #- sigma2
        risk_o = sq_loss(reg_o.predict(X_test), y_test)/len(y_test) #- sigma2
        risk_oe = sq_loss(reg_oe.predict(X_test), y_test)/len(y_test) #- sigma2

        risks_new = pd.DataFrame([[risk_2, risk_o, risk_oe]], columns = ['gd', 'achie', 'achie_e'])
        risks = pd.concat([risks, risks_new])

    # initialize plots
    fig, ax = plt.subplots()

    ax.plot(gammas, risks['gd'], 'bo', label = r'$a_{\ell_2}$')
    if include_best_achievable:
        ax.plot(gammas, risks['achie'], 'co',label = r'$a_{O}$')
    if include_best_achievable_empirical_new:
        ax.plot(gammas, risks['achie_e'], 'yo',label = r'$a_{Oe}$', markersize =4)

    ax.set_ylabel('Test error', fontsize = 'large')
    ax.set_xlabel(r'$\gamma$',fontsize = 'large')
    ax.set_title('Comparison of interpolators')
    ax.legend()

    if savefile:
        dtstamp = str(dt.now()).replace(' ', '_').replace(':','-').replace('.','_')
        folder_name = f'empir_{empir}_{dtstamp}_changing_gamma_n_{n}_r2_{r2}_sigma2_{sigma2}_ro_{str(ro)}_alpha_{str(alpha)}_regime_{regime}_alpha_{alpha}_source_{source_condition}'
        filename = 'plot.pdf'

        # make dir, save plot and params
        os.makedirs(os.path.join(plots_dir, 'rfreg', folder_name))
        fig.savefig(os.path.join(plots_dir, 'rfreg', folder_name, filename), format = 'pdf')
        with open(os.path.join(plots_dir, 'rfreg', folder_name, 'params.json'), 'w') as f:
            json.dump(saved_args, f)

    return





if __name__ == '__main__':

    display_risks_gamma_rf(n = 100,
                            r2=1,
                            sigma2=1,

                            start_gamma = 1.4,
                            end_gamma = 4,
                            gamma_2 = 3, # n/d

                            regime='id',
                            ro=0.5,

                            strong_feature_ratio = 1/2,
                            strong_feature = 1,
                            weak_feature = 1/5,

                            source_condition = 'id',

                            fix_norm_of_theta = True,

                            empir = 'gl',
                            alpha = 0.01,

                            include_best_achievable = True,

                            snr_estimation = list(np.linspace(0.1,1,20))+list(np.linspace(1,10,20)),
                            crossval_param = 100,

                            savefile = False,
                            )
