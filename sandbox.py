from __future__ import division
import numpy as np
from scipy import stats
import sympy as sp

import eeckhoutKircher2013

def _get_k_tilde(x, y, l, params):
    """Expression for k_tilde."""
    nu = params['nu']
    kappa = params['kappa']
    A = params['A']
    R = params['R']
    sp.var('k_tilde')

    # cubic coefficients
    a = -2 * R 
    b = 0.0
    c = A * kappa * nu**2 * x**0.5
    d = A * kappa * nu * (1 - nu) * (x * y * l)**0.25

    #tmp_poly = a * k_tilde**3 + b * k_tilde**2 + c * k_tilde + d
    #solns = sp.solve_poly_system([tmp_poly], k_tilde)
    #return solns[0][0]

    # discriminants
    delta = 18 * a *b * c *d - 4 * b**3 * d + b**2 * c**2 - 4 * a * c**3 - 27 * a**2 * d**2
    delta_0 = b**2 - 3 * a * c
    delta_1 = 2 * b**3 - 9 * a * b * c + 27 * a**2 * d 
    delta_2 = delta_1**2 - 4 * delta_0**3
    C = (0.5 * (delta_1 + sp.sqrt(delta_2)))**(1 / 3)

    # real roots of unity
    u_1 = 1.0 
    #u_2 = (-1 + sp.I * sp.sqrt(3)) / 2
    #u_3 = (-1 - sp.I * sp.sqrt(3)) / 2

    return -(1 / (3 * a)) * (b + u_1 * C + (delta_0 / (u_1 * C)))

def _get_capital(x, y, l, params):
    """Closed form expression for capital."""
    return _get_k_tilde(x, y, l, params)**4 

def F(x, y, l, r, params):
    """Production function."""
    nu = params['nu']
    rho = params['rho']
    A = params['A']
    kappa = params['kappa']
    
    # closed form expression for capital!
    k = _get_capital(x, y, l, params)

    y = A * kappa * (nu * (x * k)**rho + (1 - nu) * (y * l)**rho)**(nu / rho)

    return y

# distribution of land quality
mu, sigma = 0, 1
farmer_types = stats.lognorm(sigma, scale=np.exp(mu))
eps = 1e-4
x_lower, x_upper, x_measure = farmer_types.ppf(eps), farmer_types.ppf(1-eps), 0.025
farmers = eeckhoutKircher2013.Workers(farmer_types, x_lower, x_upper, x_measure)

# distribution of farmer skill
mu, sigma = -1.83, 1.0
land_prods = stats.lognorm(sigma, scale=np.exp(mu)) 
y_lower, y_upper, y_measure = land_prods.ppf(eps), land_prods.ppf(1-eps), 4.2
land = eeckhoutKircher2013.Firms(land_prods, y_lower, y_upper, y_measure)

# model parameters
params = {'nu':0.89, 'kappa':1.0, 'gamma':0.54, 'rho':0.24, 'A':1.0, 'R':0.13099}

# create an object representing the model
model = eeckhoutKircher2013.Model(params, farmers, land, F)

print model.get_wages(x_upper, [1e3, y_upper])
# compute the equilibrium using collocation
model.set_solver(solver='Shooting')
model.solver.solve(1e3, tol=1e-5, N=100, matching='pam', max_iter=2e0, 
                   integrator='lsoda', mesg=True)
