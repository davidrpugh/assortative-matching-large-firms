import sympy as sp
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, optimize, stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#from pyeconomics.models import growth
import eeckhoutKircher2013

def A(x, y, params):
    """
    Inner production function.

    Arugments:

        x: Worker type.
        y: Firm type.

    """ 
    sigma_A = params['sigma_A']
    omega_A = params['omega_A']
    
    if sigma_A == 1.0:
        out = x**omega_A * y**(1 - omega_A)
    else:
        out = ((omega_A * x**((sigma_A - 1) / sigma_A) + 
               (1 - omega_A) * y**((sigma_A - 1) / sigma_A))**(sigma_A / (sigma_A - 1))) 
    return out
 
def B(l, r, params):
    """
    Inner production function.
    
    Arguments:

        l: Quantity of labor of type x.
        r: Fraction of proprietary resources devoted to a worker
           with type x (can be normalized to unity for convenience).

    """ 
    sigma_B = params['sigma_B']
    omega_B = params['omega_B']
    
    if sigma_B == 1.0:
        out = l**omega_B * r**(1 - omega_B)
    else:
        out = ((omega_B * l**((sigma_B - 1) / sigma_B) + 
               (1 - omega_B) * r**((sigma_B - 1) / sigma_B))**(sigma_B / (sigma_B - 1)))
    return out

def F(x, y, l, r, params):
    """
    Nested CES version of production technology.

    Arguments:

        x:      worker type
        y:      firm type
        l:      quantity of labor of type x
        r:      fraction of proprietary resources devoted to a worker
                with type x (can be normalized to unity for convenience)
  
    The first two arguments x and y are quality variables describing the worker 
    and firm types, while the latter two aruments are quantity variables 
    describing the level of inputs.

    """
    return A(x, y, params) * B(l, r, params)

# describe the workers
x_lower, x_upper, x_measure = 1, 5, 1
worker_types = stats.uniform(x_lower, x_upper - x_lower) 
workers = eeckhoutKircher2013.Workers(worker_types, x_lower, x_upper, x_measure)

# describe the firms
y_lower, y_upper, y_measure = 1, 5, x_measure
firm_prods = stats.uniform(y_lower, y_upper - y_lower)
firms = eeckhoutKircher2013.Firms(firm_prods, y_lower, y_upper, y_measure)

# model parameters
params = {'omega_A':0.75, 'omega_B':0.5, 'sigma_A':0.5, 'sigma_B':1.0}

# create an object representing the model
model = eeckhoutKircher2013.Model(params, workers, firms, F)

fig, axes = plt.subplots(2, 2, figsize=(8,8))
model.set_solver(solver='Shooting')
model.solver.solve(5e0, tol=1e-5, N=1000, max_iter=1e6, integrator='lsoda', mesg=False)
model.plot_equilibrium_funcs(fig, axes, xaxis='firm_productivity')
plt.show()
