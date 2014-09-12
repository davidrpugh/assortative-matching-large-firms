import sympy as sp
import numpy as np
import pandas as pd
from scipy import integrate, interpolate, stats
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import eeckhoutKircher2013
   
# initial model parameters
params = {'omega_A':0.75, 'omega_B':0.25, 'sigma_A':0.5, 'sigma_B':1.0, 
          'alpha':1.0, 'beta':1.0}

# describe the workers
mu, sigma = 0, 1
worker_types = stats.lognorm(sigma, 0.0, np.exp(mu)) 
x_lower, x_upper, x_measure = worker_types.ppf(0.01), worker_types.ppf(0.99), 2
workers = eeckhoutKircher2013.Workers(worker_types, x_lower, x_upper, x_measure)

# describe the firms
firm_prods = stats.lognorm(sigma, 0.0, np.exp(mu)) 
y_lower, y_upper, y_measure = firm_prods.ppf(0.01), firm_prods.ppf(0.99), 1
firms = eeckhoutKircher2013.Firms(firm_prods, y_lower, y_upper, y_measure)

# create an object representing the model
model = eeckhoutKircher2013.Model(params, workers, firms, eeckhoutKircher2013.F)

# want to solve for equilibrium for different values of omega_B
sigma_As = [0.05, 0.25, 0.5, 0.75, 0.95]

# create an empty dictionary in which to store output
sigma_A_dict = {}

for val in sigma_As:
    # change value of the parameter omega_B
    model.param_dict['sigma_A'] = val
    
    # recompute partial derivatives
    model.derivatives_dict = model.compute_partialDerivatives()
    
    # dictionary of keyword arguments to pass to solver
    kwargs = {'max_order_ns':12, 'max_order_s':5, 'with_jacobian':False}

    # compute the equilibrium
    model.solve_forward_shoot(5e0, 0, h=1e-3, tol=1.5e-3, mesg=True, pandas=True, 
                             max_iter=1e6, integrator='lsoda', **kwargs)

    # use omega_B value as dict key!
    sigma_A_dict[val] = model.equilibrium
    
    print 'Done with sigma_A =', val
    
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
fig.subplots_adjust(top=0.9)

colors = ['r', 'y', 'g', 'b', 'm']

for i, val in enumerate(sigma_As):
    # extract the data
    tmp_data = sigma_A_dict[val]
    
    # generate the plots
    axes[0,0].plot(tmp_data[r'$\mu(x)$'], tmp_data[r'$\theta(x)$'], 
                   color=colors[i], label=r'$\sigma_A=%g$' %val)
    axes[0,1].plot(tmp_data.index, tmp_data[r'$\mu(x)$'], color=colors[i], 
                   label=r'$\sigma_A=%g$' %val)
    axes[1,0].plot(tmp_data[r'$\mu(x)$'], tmp_data[r'Profits, $\pi(\mu(x))$'], 
                   color=colors[i], label=r'$\sigma_A=%g$' %val)
    axes[1,1].plot(tmp_data.index, tmp_data['Wages, $w(x)$'], color=colors[i], 
                   label=r'$\sigma_A=%g$' %val)

# set plot options
axes[0,0].set_xlabel('Firm productivity, $y$')
axes[0,0].set_xlim(y_lower, y_upper)
axes[0,0].set_ylabel(r'$\theta(y)$', rotation='horizontal', fontsize=15)
axes[0,0].grid()
axes[0,0].legend(loc='best', frameon=False)

axes[0,1].set_xlabel('Worker type, $x$')
axes[0,1].set_xlim(x_lower, x_upper)
axes[0,1].set_ylabel(r'$\mu(x)$', rotation='horizontal', fontsize=15)
axes[0,1].grid()

axes[1,0].set_xlabel('Firm productivity, $y$')
axes[1,0].set_xlim(y_lower, y_upper)
axes[1,0].set_ylabel(r'$\pi(y)$', rotation='horizontal', fontsize=15)
axes[1,0].grid()

axes[1,1].set_xlabel('Worker type, $x$')
axes[1,1].set_xlim(x_lower, x_upper)
axes[1,1].set_ylabel(r'$w(x)$', rotation='horizontal', fontsize=15)
axes[1,1].grid()

fig.suptitle('Equilibria when firm productivity and worker type are ' + 
             r'$LN(%g,%g)$ and $\omega_A=%g, \omega_B=%g$' 
             %(mu, sigma, params['omega_A'], params['omega_B']), y=0.925, 
             fontsize=15)
plt.savefig('graphics/multiplicative-separability-figure2a.pdf')
plt.show()

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

for i, val in enumerate(sigma_As):
    # extract the data
    tmp_data = sigma_A_dict[val]
    
    # generate the plots
    ax.plot(tmp_data[r'$\mu(x)$'], tmp_data[r'$\theta(x)$'], color=colors[i], 
            label=r'$\sigma_A=%g$' %val)

# set plot options
ax.set_xlabel('Firm productivity, $y$ (log-scale)')
ax.set_xlim(y_lower, y_upper)
ax.set_ylabel(r'$\theta(y)$ (log-scale)', fontsize=15)
ax.set_xscale('log')
ax.set_yscale('log')
ax.grid()
ax.legend(loc='best', frameon=False)

ax.set_title('Firm size distributions are extremely skew!', fontsize=15)
plt.savefig('graphics/multiplicative-separability-figure2b.pdf')
plt.show()

