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

# empty storage container
gamma_dict = {}

for gamma in [0.5, 1.0, 2.0]:
    # distribution of worker types
    mu, sigma = 0, 1
    worker_types = stats.lognorm(sigma, 1.0, np.exp(mu)) 
    x_lower, x_upper, x_measure = worker_types.ppf(0.01), worker_types.ppf(0.99), gamma
    workers = eeckhoutKircher2013.Workers(worker_types, x_lower, x_upper, x_measure)

    # distribution of firm productivity
    firm_prods = stats.lognorm(sigma, 1.0, np.exp(mu)) 
    y_lower, y_upper, y_measure = firm_prods.ppf(0.01), firm_prods.ppf(0.99), 1
    firms = eeckhoutKircher2013.Firms(firm_prods, y_lower, y_upper, y_measure)

    # create an object representing the model
    model = eeckhoutKircher2013.Model(params, workers, firms, eeckhoutKircher2013.F)
    
    # dictionary of keyword arguments to pass to solver
    kwargs = {'max_order_ns':12, 'max_order_s':5, 'with_jacobian':False}

    # solve for the equilibrium
    model.solve_forward_shoot(2 * gamma, 0.0, h=1e-3, tol=1.5e-3, max_iter=5e5, 
                             integrator='lsoda', mesg=False, pandas=True, **kwargs)
    
    # store the equilibrium
    gamma_dict[gamma] = model.equilibrium
    
    print 'Done with gamma =', gamma    

##### Plot the results #####
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig.subplots_adjust(top=0.9)

colors = ['r', 'b', 'g']

for i, val in enumerate([0.5, 1.0, 2.0]):
    # extract the data
    tmp_data = gamma_dict[val]
    
    # generate the plots
    axes[0,0].plot(tmp_data[r'$\mu(x)$'], tmp_data[r'$\theta(x)$'], 
                   color=colors[i], label=r'$\gamma=%g$' %val)
    axes[0,1].plot(tmp_data.index, tmp_data[r'$\mu(x)$'], 
                   color=colors[i], label=r'$\gamma=%g$' %val)
    axes[1,0].plot(tmp_data[r'$\mu(x)$'], tmp_data[r'Profits, $\pi(\mu(x))$'], 
                   color=colors[i], label=r'$\gamma=%g$' %val)
    axes[1,1].plot(tmp_data.index, tmp_data['Wages, $w(x)$'], color=colors[i], 
                   label=r'$\gamma=%g$' %val)

# set plot options
axes[0,0].set_xlabel('Firm productivity, $y$')
axes[0,0].set_xlim(y_lower, y_upper)
axes[0,0].set_ylabel(r'$\theta(y)$', rotation='horizontal', fontsize=15)
axes[0,0].set_ylim(0, 3)
axes[0,0].grid()
axes[0,0].legend(loc='best', frameon=False)

axes[0,1].set_xlabel('Worker type, $x$')
axes[0,1].set_xlim(x_lower, x_upper)
axes[0,1].set_ylabel(r'$\mu(x)$', rotation='horizontal', fontsize=15)
axes[0,1].grid()

axes[1,0].set_xlabel('Firm productivity, $y$')
axes[1,0].set_xlim(y_lower, y_upper)
axes[1,0].set_ylabel(r'$\pi(y)$', rotation='horizontal', fontsize=15)
axes[1,0].set_ylim(0, y_upper)
axes[1,0].grid()

axes[1,1].set_xlabel('Worker type, $x$')
axes[1,1].set_xlim(x_lower, x_upper)
axes[1,1].set_ylabel(r'$w(x)$', rotation='horizontal', fontsize=15)
axes[1,1].set_ylim(0, x_upper)
axes[1,1].grid()

fig.suptitle(r'Equilibria when firms and workers have different population measures', 
             y=0.925, fontsize=15)
plt.savefig('graphics/multiplicative-separability-figure1.pdf')
plt.show()
