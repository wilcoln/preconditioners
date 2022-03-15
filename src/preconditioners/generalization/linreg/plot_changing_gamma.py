import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc

from preconditioners.utils import *
from preconditioners.generalization.linreg.Regressor import LinearRegressor
from datetime import datetime as dt





def display_risks_gamma(n = 100,
                        r2=1,
                        sigma2=1,

                        start_gamma = 1.4,
                        end_gamma = 4,

                        regime='exponential',
                        ro=0.5,

                        strong_feature_ratio = 1/2,
                        strong_feature = 1,
                        weak_feature = 1/5,

                        source_condition = 'id',

                        empir = 'variance_gl',
                        alpha = 0.25,

                        include_gd = False,
                        include_md = False,
                        include_md_empirical = False,
                        include_best_achievable = True,
                        include_best_achievable_empirical_gl = True,
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

        Smallest value of gamma (n/d) in the plot.

    end_gamma : float > 0

        Largest value of gamma (n/d) in the plot.

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

    empir : {'gl', 'basic', 'lw'}

        Specifies what kind of approximation of the population covariance matrix we use. 'gl' is the GraphicalLasso
        approximation, 'basic' is the standard X^TX/n, 'lw' is the LedoitWolf approximation.

    alpha : float > 0

        Only used if empir = 'gl'. Specifies the regularization of the GraphicalLasso approximation.
        This can be also crossvalidated, using GraphicalLassoCV from sklearn.

    include_gd : Boolean

        If true then includes the minimum norm interpolator in the plots.

    include_md : Boolean

        If true then includes the best variance interpolator (covariance mirror descent initialized at 0) in the plots.

    include_md_empirical : Boolean

        If true then includes the empirical approximation of the best variance interpolator.

    include_best_achievable : Boolean

        If true then includes the best linearly achievable interpolator in the plots.

    include_best_achievable_empirical_new : Boolean

        If true then includes the empirical approximation of the best linearly achievable interpolator in the plots
        using the Graphical Lasso estimator of the covariance matrix.

    include_best_achievable_empirical_new : Boolean

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

    snr = r2/sigma2

    # generate sequence of gammas for plotting
    gammas = np.concatenate((np.linspace(start_gamma,start_gamma+(end_gamma-start_gamma)/3,8),np.linspace(start_gamma+(end_gamma-start_gamma)/3+(end_gamma-start_gamma)/15,end_gamma,7)))

    risks = np.zeros((len(gammas), 7))

    count = 0
    # do experiment for each gamma
    for i in range(len(gammas)):
        count = count+1
        print(f'{count}/15')

        gamma = gammas[i]
        d = int(gamma*n)

        # generate covariance matrix of data
        c = generate_c(ro = ro, regime = regime, n = n, d = d,
                        strong_feature = strong_feature,
                        strong_feature_ratio = strong_feature_ratio,
                        weak_feature = weak_feature)

        # generate covariance matrix of the prior of the true parameter
        m = generate_m(c, source_condition = source_condition)

        # generate true parameter
        w_star = generate_true_parameter(n, d, r2, m = m)

        # generate data
        X, y, xi = generate_centered_gaussian_data(w_star, c, n, d, sigma2)
        print('data generated')

        # generate empirical estimate of the covariance matrix
        c_e = generate_c_empir(X, empir, alpha)
        print('covariance matrix estimated')

        # initialize models
        reg_2 = LinearRegressor()
        reg_c = LinearRegressor()
        reg_ce = LinearRegressor()

        # generate predictors
        # matrix specifies which mirror descent we are using (GD if None)
        reg_2.fit(X, y, matrix = None)
        reg_c.fit(X, y, matrix = c)
        reg_ce.fit(X, y, matrix = c_e)

        w_a = compute_best_achievable_interpolator(X, y, c, m, snr)
        reg_a = LinearRegressor(init = w_a)

        if include_best_achievable_empirical_new:
            w_ae = compute_best_achievable_interpolator(X, y, c = c_e, m = np.eye(d), snr = snr_estimation, crossval_param = crossval_param)
            reg_ae = LinearRegressor(init = w_ae)
            print('empirical approximation computed')
        
        if include_best_achievable_empirical_gl:
            c_gl = generate_c_empir(X, 'gl', alpha)
            w_gl = compute_best_achievable_interpolator(X, y, c = c_gl, m = np.eye(d), snr = snr_estimation, crossval_param = crossval_param)
            reg_gl = LinearRegressor(init = w_gl)
            print('empirical approximation computed')

        # best possible linear predictor
        c_mhalf = np.linalg.inv(sc.sqrtm(c)) # inverse square root of the covariance matrix
        w_b = c_mhalf.dot( np.linalg.lstsq( X.dot(c_mhalf),  xi, rcond=None)[0] ) + w_star # best possible predictor
        reg_b = LinearRegressor(init = w_b)

        # calculate the expected risks
        risk_2 = calculate_risk(w_star, c, reg_2.w ) + sigma2
        risk_c = calculate_risk(w_star, c, reg_c.w) + sigma2
        risk_ce = calculate_risk(w_star, c, reg_ce.w) + sigma2
        risk_a = calculate_risk(w_star, c, reg_a.w) + sigma2
        if include_best_achievable_empirical_new:
            risk_ae = calculate_risk(w_star, c, reg_ae.w) + sigma2
        if include_best_achievable_empirical_gl
            risk_gl = calculate_risk(w_star, c, reg_gl.w) + sigma2
        risk_b = calculate_risk(w_star, c, reg_b.w) + sigma2

        risks[i, :] = risk_2, risk_c, risk_ce, risk_a, risk_ae, risk_gl, risk_b


    # initialize plots
    fig, ax = plt.subplots()

    if include_gd:
        ax.plot(gammas, risks[:, 0], 'bo', label = r'$w_{\ell_2}$')
    if include_md:
        ax.plot(gammas, risks[:, 1], 'ro', label = r'$w_{V}$')
    if include_md_empirical:
        ax.plot(gammas, risks[:, 2], 'mo',label = r'$w_{Ve}$', markersize = 4)
    if include_best_achievable:
        ax.plot(gammas, risks[:, 3], 'co',label = r'$w_{O}$')
    if include_best_achievable_empirical_new:
        ax.plot(gammas, risks[:, 4], 'yo',label = r'$w_{Oe}$', markersize = 4)
    if include_best_achievable_empirical_gl:
        ax.plot(gammas, risks[:, 4], 'go',label = r'$w_{Ogl}$', markersize = 4)
    ax.plot(gammas, risks[:, 5], 'ko', label = r'$w_{b}$')


    ax.set_ylabel('Risk', fontsize = 'large')
    ax.set_xlabel(r'$\gamma$',fontsize = 'large')
    ax.set_title('Comparison of interpolators')
    ax.legend()

    if savefile:
        dtstamp = str(dt.now()).replace(' ', '_').replace(':','-').replace('.','_')
        fig.savefig(f'images/changing_gamma_n_{n}_r2_{r2}_sigma2_{sigma2}_ro_{str(ro)}_alpha_{str(alpha)}_regime_{regime}_alpha_{alpha}_source_{source_condition}_final_{dtstamp}.pdf', format = 'pdf')

    return





if __name__ == '__main__':

    display_risks_gamma(n = 100,
                        r2=1,
                        sigma2=1,

                        start_gamma = 1.4,
                        end_gamma = 4,

                        regime='exponential',
                        ro=0.5,

                        strong_feature_ratio = 1/2,
                        strong_feature = 1,
                        weak_feature = 1/5,

                        source_condition = 'id',

                        empir = 'gl',
                        alpha = 0.25,

                        include_gd = False,
                        include_md = False,
                        include_md_empirical = False,
                        include_best_achievable = True,
                        include_best_achievable_empirical_gl = True,
                        include_best_achievable_empirical_new = True,

                        snr_estimation = list(np.linspace(0.1,1,20))+list(np.linspace(1,10,20)),
                        crossval_param = 100,

                        savefile = False,
                        )
