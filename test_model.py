"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-09-30

"""

        
        
if __name__ == '__main__':
    import numpy as np
    from scipy import stats

    from inputs import Input

    params = {'nu':0.89, 'kappa':1.0, 'gamma':0.54, 'rho':0.24, 'A':1.0}

    sym.var('A, kappa, nu, x, rho, y, l, gamma, r')
    F = r * A * kappa * (nu * x**rho + (1 - nu) * (y * (l / r))**rho)**(gamma / rho)

    # suppose worker skill is log normal
    sp.var('x, mu, sigma')
    skill_cdf = 0.5 + 0.5 * sp.erf((sp.log(x) - mu) / sp.sqrt(2 * sigma**2))
    worker_params = {'mu':0.0, 'sigma':1}
    x_lower, x_upper, x_measure = 1e-3, 5e1, 0.025
    
    workers = Input(distribution=skill_cdf,
                    lower_bound=x_lower,
                    params=worker_params,
                    upper_bound=x_upper,
                   )

    # suppose firm productivity is log normal
    sp.var('x, mu, sigma')
    productivity_cdf = 0.5 + 0.5 * sp.erf((sp.log(x) - mu) / sp.sqrt(2 * sigma**2))
    firm_params = {'mu':0.0, 'sigma':1}
    y_lower, y_upper, y_measure = 1e-3, 5e1, 0.025
    
    firms = Input(distribution=productivity_cdf,
                  lower_bound=x_lower,
                  params=firm_params,
                  upper_bound=x_upper,
                  )

    # create an instance of the Model class
    model = Model(firms=firms,
                  matching='PAM',
                  output=F,
                  params=params,
                  workers=workers
                  )

    print(model._numeric_jacobian(x_upper, 1e2, y_upper))
    print(model._numeric_ode_system(x_upper, 1e2, y_upper))
