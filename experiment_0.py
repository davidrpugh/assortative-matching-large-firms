import sympy as sp
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import eeckhoutKircher2013

########## Varying the relative measures of workers and firms ##########

# model parameters
params = {'omega_A':0.5, 'omega_B':0.5, 'sigma_A':0.5, 'sigma_B':1.0, 
          'alpha':1.0, 'beta':1.0}

# distribution of worker types and firm productivity is lognormal
mu, sigma = 0, 1
common_dist = stats.lognorm(sigma, 0.0, np.exp(mu)) 
lower, upper, measure = common_dist.ppf(0.01), common_dist.ppf(0.99), 1
 
workers = eeckhoutKircher2013.Workers(common_dist, lower, upper, measure)
firms = eeckhoutKircher2013.Firms(common_dist, lower, upper, measure)

# create an object representing the model
model = eeckhoutKircher2013.Model(params, workers, firms, eeckhoutKircher2013.F)
    
# dictionary of keyword arguments to pass to solver
kwargs = {'max_order_ns':12, 'max_order_s':5, 'with_jacobian':False}

# solve for the equilibrium
model.solve_forward_shoot(1.0, tol=1.5e-3, N=100, max_iter=5e5, 
                          integrator='lsoda', mesg=False, pandas=True, **kwargs)
    
##### Plot the results #####
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig.subplots_adjust(top=0.9)
    
# generate the plots
axes[0,0].plot(model.equilibrium[r'$\mu(x)$'], 
               model.equilibrium[r'$\theta(x)$'])
axes[0,1].plot(model.equilibrium.index, model.equilibrium[r'$\mu(x)$'])
axes[1,0].plot(model.equilibrium[r'$\mu(x)$'], 
               model.equilibrium[r'Profits, $\pi(\mu(x))$'])
axes[1,1].plot(model.equilibrium.index, model.equilibrium['Wages, $w(x)$'])

# set plot options
axes[0,0].set_xlabel('Firm productivity, $y$')
axes[0,0].set_xlim(lower, upper)
axes[0,0].set_ylabel(r'$\theta(y)$', rotation='horizontal', fontsize=15)
axes[0,0].set_ylim(0, 3)
axes[0,0].grid()

axes[0,1].set_xlabel('Worker type, $x$')
axes[0,1].set_xlim(lower, upper)
axes[0,1].set_ylabel(r'$\mu(x)$', rotation='horizontal', fontsize=15)
axes[0,1].autoscale(tight=True)
axes[0,1].grid()

axes[1,0].set_xlabel('Firm productivity, $y$')
axes[1,0].set_xlim(lower, upper)
axes[1,0].set_ylim(0, upper)
axes[1,0].set_ylabel(r'$\pi(y)$', rotation='horizontal', fontsize=15)
axes[1,0].grid()

axes[1,1].set_xlabel('Worker type, $x$')
axes[1,1].set_xlim(lower, upper)
axes[1,1].set_ylim(0, upper)
axes[1,1].set_ylabel(r'$w(x)$', rotation='horizontal', fontsize=15)
axes[1,1].grid()

fig.suptitle('Equilibrium under complete symmetry assumptions', y=0.925, 
             fontsize=20, weight='bold')
plt.savefig('graphics/multiplicative-separability-figure0.pdf')
plt.show()
