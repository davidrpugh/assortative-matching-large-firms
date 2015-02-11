from __future__ import division
import numpy as np
from numpy.polynomial import Chebyshev 
from scipy import interpolate, optimize, stats

import matplotlib as mpl
import matplotlib.pyplot as plt

class Workers(object):
    """Abstract class representing workers in Eeckhout and Kircher (2013)."""

    def __init__(self, dist, lower_bound, upper_bound, size, **kwargs):
        """
        Initializes a Workers object with the following attributes:

            dist:        (rv_frozen) distribution of worker types (must be from 
                         scipy.stats!).
            
            lower_bound: (float) lower bound on the feasible range of values for
                         a worker's type. If the distribution of worker types
                         has finite support, then this is simply the lower bound
                         of the support; if the distribution has unbounded
                         support then this should be set to some quantile of the
                         cdf. 
            
            upper_bound: (float) upper bound on the feasible range of values for
                         a worker's type. If the distribution of worker types
                         has finite support, then this is simply the upper bound
                         of the support; if the distribution has unbounded
                         support then this should be set to some quantile of the
                         cdf.   
            
            size:        (float) size is the measure of workers.

        Optional keyword arguments:

            'unemployment_insurance': (Function) Unemployment insurance. In this
                                      model unemployment insurance is some 
                                      function of worker type. Default behavior 
                                      is to define

                                          unemployment_insurance = lambda x: 0

                                      which implies that there is no insurance
                                      for workers of any type. 

        """
        self.dist        = dist
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound 
        self.size        = size

        # function form for unemployment insurance
        self.unemployment_insurance = kwargs.get('unemployment_insurance', 
                                                 lambda x: 0)

        # normalization constant for pdf
        self.norm_const = (self.dist.cdf(self.upper_bound) - 
                           self.dist.cdf(self.lower_bound)) 
        
        # define the scaling factor for pdf
        self.scaling_factor = self.size / self.norm_const
        
    def scaled_pdf(self, xn):
        """Scaled probability density function for worker type/skill."""
        return self.scaling_factor * self.dist.pdf(xn)

class Firms(object):
    """Abstract class representing firms in Eeckhout and Kircher (2013)."""

    def __init__(self, dist, lower_bound, upper_bound, size, **kwargs):
        """
        Initializes a Firms object with the following attributes:

            dist:        (rv_frozen) distribution of firm productivities (must 
                         be from scipy.stats!).
            
            lower_bound: (float) lower bound on the feasible range of values for
                         a firm's productivity. If the distribution of firm
                         productivity has finite support, then this is simply
                         the lower bound of the support; if the distribution
                         has unbounded support then this should be set to some 
                         quantile of the cdf. 
            
            upper_bound: (float) upper bound on the feasible range of values for
                         a firm's productivity. If the distribution of firm
                         productivity has finite support, then this is simply
                         the lower bound of the support; if the distribution
                         has unbounded support then this should be set to some 
                         quantile of the cdf.   
            size:        (float) size is the measure of firms.

        Optional keyword arguments:

            'fixed_costs': (function) Equilibrium fixed costs for firms are 
                           match specific? Default behavior is to define
                           
                               fixed_costs = lambda x, vec: 0

                           which implies that there are no fixed costs for any 
                           firm regardless of productivity.
                           
        """
        self.dist        = dist
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound 
        self.size        = size 

        # functional form for match specific fixed costs
        self.fixed_costs = kwargs.get('fixed_costs', lambda x, vec: 0)

        # normalization constant for pdf
        self.norm_const = (self.dist.cdf(self.upper_bound) - 
                           self.dist.cdf(self.lower_bound)) 
        
        # define the scaling factor for pdf
        self.scaling_factor = self.size / self.norm_const
        
    def scaled_pdf(self, xn):
        """Scaled probability density function for firm productivity."""
        return self.scaling_factor * self.dist.pdf(xn) 

# defining the production function
def A(x, y, params):
    """CES between x and y.""" 
    omega_A = params['omega_A']
    sigma_A = params['sigma_A']
    
    rho = (sigma_A - 1) / sigma_A
    out = (omega_A * x**rho + (1 - omega_A) * y**rho)**(1 / rho)
    
    return out

def Ax(x, y, params):
    """Partial derivative of A(x,y) with respect to worker skill, x.""" 
    omega_A = params['omega_A']
    sigma_A = params['sigma_A']
    
    rho = (sigma_A - 1) / sigma_A
    out = (omega_A * x**(rho - 1) * 
           (omega_A * x**rho + (1 - omega_A) * y**rho)**((1 / rho) - 1)) 
    
    return out

def Ay(x, y, params):
    """Partial derivative of A(x,y) with respect to firm productivity, y.""" 
    omega_A = params['omega_A']
    sigma_A = params['sigma_A']
    
    rho = (sigma_A - 1) / sigma_A    
    out = ((1 - omega_A) * y**(rho - 1) * 
           (omega_A * x**rho + (1 - omega_A) * y**rho)**((1 / rho) - 1)) 
    
    return out

def Axy(x, y, params):
    """Cross partial of Ax with respect to y."""
    omega_A = params['omega_A']
    sigma_A = params['sigma_A']
    
    rho = ((sigma_A - 1) / sigma_A)     
    out = ((1 - rho) * omega_A * (1 - omega_A) * (x * y)**(rho - 1) * 
           ((omega_A * x**rho + (1 - omega_A) * y**rho)**((1 / rho) - 2))) 
    
    return out
    
def B(l, r, params):
    """Cobb-Douglas between l and r"""  
    omega_B = params['omega_B']   
    
    out = l**omega_B * r**(1 - omega_B) 
    
    return out

def Bl(l, r, params):
    """Partial derivative of B(l, r) with respect to quantity of labor, l."""  
    omega_B = params['omega_B']    
    
    out = omega_B * (l / r)**(omega_B - 1)
    
    return out

def Br(l, r, params):
    """Partial derivative of B(l, r) with respect to firm resources, r."""     
    omega_B = params['omega_B']   
    
    out = (1 - omega_B)  * l**omega_B * r**(-omega_B) 
    
    return out

def Blr(l, r, params):
    """Cross partial derivative of Bl(x, vec, params) with respect to r."""    
    omega_B = params['omega_B']   
    
    out = omega_B * (1 - omega_B)  * l**(omega_B - 1) * r**(-omega_B)
     
    return out

def F(x, y, l, r, params):
    """Multiplicitively separable production technology."""
    return A(x, y, params) * B(l, r, params)

def Fyl(x, y, l, r, params):
    """Span-of-control complementarity."""
    out = Ay(x, y, params) * Bl(l, r, params)
    return out

def Fxr(x, y, l, r, params):
    """Managerial resources complementarity."""
    out = Ax(x, y, params) * Br(l, r, params)
    return out

def Flr(x, y, l, r, params):
    """Workers resources complementarity."""
    out = A(x, y, params) * Blr(l, r, params) 
    return out

def Fxy(x, y, l, r, params):
    """Type complementarity."""
    out = Axy(x, y, params) * B(l, r, params)
    return out

def H(x, y):
    """
    Ratio of density functions of worker skill and firm productivity 
    (evaluated along the equilibrium path!).

    Arguments:

        xn:     (float) Worker type/skill.
            
    Returns:
            
        out: (array) Ratio of worker-firm densities.

    """
    out = workers.scaled_pdf(x) / firms.scaled_pdf(y)
    return out

"""
# distribution of worker types
x_lower, x_upper, x_measure = 1.0, 50.0, 1.0
worker_types = stats.uniform(x_lower, x_upper - x_lower) 
workers = Workers(worker_types, x_lower, x_upper, x_measure)

# distribution of firm productivity
y_lower, y_upper, y_measure = 1.0, 50.0, 1.0
firm_prods = stats.uniform(y_lower, y_upper - y_lower)
firms = Firms(firm_prods, y_lower, y_upper, y_measure)
"""

mu, sigma = 0, 1
worker_types = stats.lognorm(sigma, 0.0, np.exp(mu)) 
x_lower, x_upper, x_measure = worker_types.ppf(0.01), worker_types.ppf(0.99), 1
workers = Workers(worker_types, x_lower, x_upper, x_measure)

# describe the firms
firm_prods = stats.lognorm(sigma, 0.0, np.exp(mu)) 
y_lower, y_upper, y_measure = firm_prods.ppf(0.01), firm_prods.ppf(0.99), 1
firms = Firms(firm_prods, y_lower, y_upper, y_measure)

def get_pam_system(x, mu, theta, params):
    """
    2D system of differential equations describing the behavior of an 
    equilibrium with positive assortative matching between workers and 
    firms.

    Arguments:

        x:      (array-like) Worker type/skill.
        mu:     (array-like) Productivity of firm matched with type x worker.
        theta:  (array-like) Factor intensity (size) of firm matched with
                worker of type x.
        params: (dict) Dictionary of model parameters.
                
    Returns:
            
        out: (array) Vector of values [mu', theta'].

    """
    out = np.hstack((mu_prime(x, mu, theta, params), 
                     theta_prime(x, mu, theta, params)))
    return out

def theta_prime(x, mu, theta, params):
    """ODE governing the evolution of factor intensity."""
    out = ((H(x, mu) * Fyl(x, mu, theta, 1, params) - Fxr(x, mu, theta, 1, params)) / 
           Flr(x, mu, theta, 1, params))               
    return out

def mu_prime(x, mu, theta):
    """ODE governing the evolution of matching function."""
    return H(x, mu) / theta
             
def theta_hat(coefs, x_lower, x_upper):
    """
    Chebyshev polynomial approximation of theta(x).
    
    Arguments:
        
        coefs:       (array-like) Chebyshev polynomial coefficients.
                
        x_lower:     (float) Lower bound on support of the distribution of 
                     worker type.
        
        x_upper:     (float) Upper bound on support of the distribution of 
                     worker type.
        
    Returns:
        
        poly: (object) Instance of the Chebeyshev class.
        
    """
    poly = Chebyshev(coefs, [x_lower, x_upper])
    return poly

def mu_hat(coefs, x_lower, x_upper):
    """
    Chebyshev polynomial approximation of mu(x).
    
    Arguments:
        
        coefs:       (array-like) Chebyshev polynomial coefficients.
                
        x_lower:     (float) Lower bound on support of the distribution of 
                     worker type.
        
        x_upper:     (float) Upper bound on support of the distribution of 
                     worker type.
        
    Returns:
        
        poly: (object) Instance of the Chebeyshev class.
        
    """
    poly = Chebyshev(coefs, [x_lower, x_upper])
    return poly
       
def R(x, coefs, x_lower, x_upper, params):
    """Residual function.
    
    Arguments:
        
        coefs:   (array-like) (2, m) array of Chebyshev polynomial 
                 coefficients. The first row of coefficients is used to 
                 construct the mu_hat(x) approximation of the equilibrium
                 matching function mu(x). The second row of coefficients is
                 used to construct the theta_hat(x) approximation of the
                 equilibrium firm size function theta(x).
                     
        x_lower: (float) Lower bound on support of the distribution of 
                 worker type.
        
        x_upper: (float) Upper bound on support of the distribution of 
                 worker type.
                     
        params:  (dict) Dictionary of model parameters.  
                     
    """
    
    # construct the polynomial approximations of mu(x) and theta(x)
    mu      = mu_hat(coefs[0], x_lower, x_upper)
    mu_x    = mu.deriv()
    
    theta   = theta_hat(coefs[1], x_lower, x_upper)
    theta_x = theta.deriv()
        
    # compute the residual polynomial
    res_mu    = mu_x(x) - mu_prime(x, mu(x), theta(x))
    res_theta = theta_x(x) - theta_prime(x, mu(x), theta(x), params)
          
    res = np.hstack((res_mu, res_theta)) 
    return res

def collocation_system(coefs, nodes, x_lower, x_upper, params):
    """
    System of non-linear equations for collocation method.
    
    Arguments:
        
        coefs:   (array-like) (2, m) array of Chebyshev polynomial 
                 coefficients. The first row of coefficients is used to 
                 construct the mu_hat(x) approximation of the equilibrium
                 matching function mu(x). The second row of coefficients is
                 used to construct the theta_hat(x) approximation of the
                 equilibrium firm size function theta(x).
              
        nodes:   (array) (m + 1,) array of Chebyshev roots used as collocation
                 nodes.
               
        x_lower: (float) Lower bound on support of the distribution of 
                 worker type.
        
        x_upper: (float) Upper bound on support of the distribution of 
                 worker type.
                     
        params:  (dict) Dictionary of model parameters.  
        
    Returns:
        
        out: (array) Array of residuals.
        
    """
    coefs = coefs.reshape((2, m + 1))
    
    # compute the values of the boundary conditions
    mu = mu_hat(coefs[0], x_lower, x_upper)
    bc = np.array([mu(x_lower) - firms.lower_bound, 
                   mu(x_upper) - firms.upper_bound])
    
    out = np.hstack((bc, R(nodes, coefs, x_lower, x_upper, params)))
    return out
    
params = {'omega_A':0.55, 'omega_B':0.5, 'sigma_A':0.75}

# desired degree of polynomial
m = 60
x_lower, x_upper = workers.lower_bound, workers.upper_bound

init_mu_coefs = np.zeros(m + 1)
init_mu_coefs[0] = 0.5 * (x_upper + x_lower)
init_mu_coefs[1] = 0.5 * (x_upper - x_lower)

init_theta_coefs = np.zeros(m + 1)
init_theta_coefs[0] = 1

init_coefs = np.vstack((init_mu_coefs, init_theta_coefs))
tmp_coefs = init_coefs 

# collocation nodes are Chebyshev roots    
basis_coefs     = np.zeros(m + 1)
basis_coefs[-1] = 1
nodes           = Chebyshev(basis_coefs, [x_lower, x_upper]).roots()
print nodes.size    
# empty dictionaries for storing output
results_dict = {}

omega_As = np.linspace(0.50, 0.01, 25)
     
for i, omega_A in enumerate(omega_As):
    
    # change the parameter
    params['omega_A'] = omega_A
       
    # solve the system of non-linear equations 
    tmp_res = optimize.root(collocation_system, tmp_coefs.flatten(), 
                            args=(nodes, x_lower, x_upper, params), 
                            method='hybr')
    
    # store the results
    results_dict[omega_A] = tmp_res
        
    # hot start for next value of omega_A
    tmp_coefs = tmp_res.x.reshape((2, m + 1))

    print 'Done with omega_A = %g' % omega_A

fig = plt.figure(figsize=(8,6))
ax0 = fig.add_subplot(311) 
ax1 = fig.add_subplot(312) 
ax2 = fig.add_subplot(313)

plot_grid = np.linspace(x_lower, x_upper, 1000)

colors = mpl.cm.jet(np.linspace(0, 1, 50))

omega_As = np.linspace(0.50, 0.01, 25)
     
for i, omega_A in enumerate(omega_As):
    
    # extract the results dictionary
    tmp_res = results_dict[omega_A]
           
    # store callable mu(x) and theta(x)
    tmp_mu    = mu_hat(tmp_res.x[:m + 1], x_lower, x_upper)
    tmp_theta = theta_hat(tmp_res.x[m + 1:], x_lower, x_upper)
        
    mu_vals    = tmp_mu(plot_grid)
    theta_vals = tmp_theta(plot_grid)
    
    ax0.plot(plot_grid, mu_vals, color=colors[25 - i])
    ax1.plot(plot_grid, theta_vals, color=colors[25 - i])
    
    # inverse of x(theta)
    x_theta = interpolate.InterpolatedUnivariateSpline(theta_vals, 
                                                       plot_grid, 
                                                       k=1, bbox=[1e-4, 1e3])
    
    grid = np.linspace(theta_vals.min(), theta_vals.max(), 1000)
    
    ax2.plot(grid, workers.dist.pdf(x_theta(grid)) / (grid.max() - grid.min()), 
             color=colors[25 - i])
   
ax0.set_ylabel('$\mu(x)$', rotation='horizontal')

ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$\theta(x)$', rotation='horizontal')
ax1.set_yscale('log')
ax1.set_xscale('log')
#ax2.set_ylim(0, 4) 
ax2.set_xscale('log')
ax2.set_yscale('log')
fig.tight_layout()
plt.show()

"""


for m in range(1, M):
    
    # collocation nodes are Chebyshev roots    
    tmp_basis_coefs     = np.zeros(m + 1)
    tmp_basis_coefs[-1] = 1
    tmp_grid            = Chebyshev(tmp_basis_coefs, [x_lower, x_upper]).roots()

    # solve the system of non-linear equations 
    tmp_res = optimize.root(collocation_system, tmp_coefs, 
                            args=(tmp_grid, x_lower, x_upper, params), 
                            method='hybr')

    # create Chebyshev polynomials using the results
    tmp_mu    = mu_hat(tmp_res.x[:m + 1], x_lower, x_upper)
    tmp_theta = theta_hat(tmp_res.x[m + 1:], x_lower, x_upper)

    ax0.plot(plot_grid, tmp_mu(plot_grid), color=colors[m - 1])
    ax0.set_ylabel('$\mu(x)$', rotation='horizontal')

    ax1.plot(plot_grid, tmp_theta(plot_grid), color=colors[m - 1])
    ax1.set_xlabel('$x$')
    ax1.set_ylabel(r'$\theta(x)$', rotation='horizontal')
    
    tmp_coefs = np.vstack((np.append(tmp_mu.coef, 0), np.append(tmp_theta.coef, 0)))
    tmp_coefs = tmp_coefs.flatten()
    
fig.tight_layout()
plt.show()

##### Residual plot #####

fig = plt.figure(figsize=(8,6))
ax0 = fig.add_subplot(211) 
ax1 = fig.add_subplot(212) 

final_coefs = np.vstack((tmp_mu.coef, tmp_theta.coef))

N = 1000
R_mu = R(plot_grid, final_coefs, x_lower, x_upper, params)[:N]
R_theta = R(plot_grid, final_coefs, x_lower, x_upper, params)[N:]

plot_grid = np.linspace(x_lower, x_upper, N)

ax0.plot(plot_grid, R_mu[:N], color=colors[m - 1])
ax0.set_xlabel('$x$')
ax0.set_ylabel(r'$R_{\mu}$', rotation='horizontal')

ax1.plot(plot_grid, R_theta[:N], color=colors[m - 1])
ax1.set_xlabel('$x$')
ax1.set_ylabel(r'$R_{\theta}$', rotation='horizontal')

plt.show()
"""