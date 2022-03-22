import numpy as np
from preconditioners.generalization.rfreg.plot_changing_gamma_random_features import display_risks_gamma_rf




if __name__ == '__main__':


    # for smaller run time (but noisier plot) decrease n
    display_risks_gamma_rf(n = 100,
                        n_extra = 1,
                        n_benchmark = 3000,
                        n_test = 200,
                        r2=1,
                        sigma2=1,

                        start_gamma = 4,
                        end_gamma = 7,
                        gamma_2 = 3, # n/d

                        regime='id',
                        ro=0.5,

                        source_condition = 'id',

                        fix_norm_of_theta = True,
                        fix_norm_of_x = True,

                        empir = 'test',
                        alpha = 0.2,
                        mu = 0,
                        geno_tol = 1e-5,

                        include_best_achievable = True,
                        include_best_achievable_empirical_new=True,

                        snr_estimation = 1,
                        crossval_param = 1,

                        savefile = True,
                        )

