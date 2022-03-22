import numpy as np
from preconditioners.generalization.linreg.plot_changing_gamma import display_risks_gamma




if __name__ == '__main__':


    # for smaller run time (but noisier plot) decrease n
    display_risks_gamma(n = 100,
                        r2=1,
                        sigma2=5,

                        start_gamma = 1.4,
                        end_gamma = 4,

                        regime='autoregressive',
                        ro=0.5,

                        source_condition = 'id',

                        empir = 'variance_gl',
                        alpha = 0.1,
                        geno_tol=1e-4,

                        include_gd = True,
                        include_md = False,
                        include_md_empirical = False,
                        include_best_achievable = True,
                        include_best_achievable_empirical_new = True,
                        include_best_achievable_empirical_gl = True,

                        snr_estimation = 1,
                        crossval_param = 10,

                        savefile = True,
                        )
