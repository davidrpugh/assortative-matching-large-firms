from __future__ import division
import numpy as np
from numpy.polynomial import Chebyshev, Legendre, Laguerre, Hermite
from scipy import optimize, stats
import sympy as sp
import pandas as pd
from scipy import integrate, interpolate
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
        
    def scaled_pdf(self, x):
        """Scaled probability density function for worker type/skill."""
        return self.scaling_factor * self.dist.pdf(x)
        
    def scaled_cdf(self, x):
        """Scaled probability distribution function for worker type/skill."""
        cdf = (self.scaling_factor * (self.dist.cdf(x) - 
                                      self.dist.cdf(self.lower_bound)))
        return cdf 
        
    def scaled_sf(self, x):
        """Scaled survival function for worker type/skill."""
        return 1 - self.scaled_cdf(x)
        
    def rvs(self, N):
        """Draws from scaled distribution for worker type/skill."""
        q     = stats.uniform.rvs(0, 1, N)
        draws = self.dist.ppf((q / self.scaling_factor) + 
                              self.dist.cdf(self.lower_bound))
        return draws

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
        
    def scaled_pdf(self, x):
        """Scaled probability density function for firm productivity."""
        return self.scaling_factor * self.dist.pdf(x)
        
    def scaled_cdf(self, x):
        """Scaled probability distribution function for firm productivity."""
        cdf = (self.scaling_factor * (self.dist.cdf(x) - 
                                      self.dist.cdf(self.lower_bound)))
        return cdf
        
    def scaled_sf(self, x):
        """Scaled survival function for firm productivity."""
        return 1 - self.scaled_cdf(x)
        
    def rvs(self, N):
        """Draws from scaled distribution for firm productivity."""
        q     = stats.uniform.rvs(0, 1, N)
        draws = self.dist.ppf((q / self.scaling_factor) + 
                              self.dist.cdf(self.lower_bound))
        return draws
        
class Model(object):
    """
    Class for solving and simulating the equilibrium of the Eeckhout and 
    Kircher (2013) model.

    ABSTRACT: In large firms, management resolves a trade-off between hiring
    more versus better workers.  The span of control or size if therefore 
    intimately intertwined with the sorting pattern. Span of control is at the 
    center of many studies in macroeconomics, comparisons of factor 
    productivity, trade, and labor.  With hetergeneous workers, we analyze the 
    worker assignment, firm size, and wages.  The pattern of sorting between 
    workers and firms is governed by an intuitive cross-margin-complementarity 
    condition that captures the complementarities between qualities (of workers 
    and firms), and quantities (of the work force and firm resources). A simple 
    system of two differential equations determines the equilibrium allocation, 
    firm size, and wages.  We can analyze the impact of technological change: 
    skill-biased change affects wages and increases the skill premium; 
    quantity-biased change affects firm size, especially of productive firms.  
    We also introduce search frictions an investigate how unemployment varies 
    across skills and how vacancies vary across firm size.

    """
    def __init__(self, params, workers, firms, F):
        """
        Initializes an object representing the Eeckhout and Kircher (2013) 
        model of assortative matching with large firms.

        Attributes:

            params:  (dict) Dictionary of model parameters.
            
            workers: (object) An object of the Workers class.
            
            firms:   (object) An object of the Firms class.
            
            F:       (function) Output, F, is a function of worker type, x,
                     firm productivity, y, quantity of labor of type x, l, and
                     fraction of proprietary resources devoted to a worker with
                     type x, r. Final argument of F should be a dictionary of 
                     model parameters.

        """
        self.params  = params.copy()
        self.workers = workers
        self.firms   = firms        
        self.F       = F
        
        # compute the partial and cross-partial derivatives
        self.derivatives_dict = self.compute_partial_derivatives()
        
        # model solver
        self.solver = None
        
        # model equilibrium will be a dictionary of callable funcs
        self.equilibrium = {}
                
    def check_pam(self, xn, vec):
        """
        Checks that the necessary condition required for a positive 
        assortative matching equilibrium.  See Theorem 1 from the paper for 
        details. This condition must be satisfied at each step along the 
        equilibrium path.

        Arguments:

            xn:  worker type/skill
            vec: vector of variables, vec = [theta, mu]

        Returns:

            check: (boolean) True if PAM, False if NAM.
        
        """
        # catch case where vec element is NaN
        if np.isnan(vec).any() == True:
            check = True
        
        # defines the inequality from equation 12
        else:
            LHS = (self.derivatives_dict['Fxy'](xn, vec[1], vec[0], 1) * 
                   self.derivatives_dict['Flr'](xn, vec[1], vec[0], 1))
            RHS = (self.derivatives_dict['Fyl'](xn, vec[1], vec[0], 1) * 
                   self.derivatives_dict['Fxr'](xn, vec[1], vec[0], 1)) 

            # if positive then PAM, otherwise NAM!
            check = (LHS >= RHS)

        return check

    def check_proposition_2(self, xn, vec, matching='positive'):
        """
        If matching is assortative and differentiable, and output is 
        increasing in worker types, better firms will hire more workers if and
        only if along the equilibrium path:

            H(x)Fyl < Fxr under PAM
            H(x)Fyl > -Fxr under NAM

        See proposition 2 from the paper for the details.

        Arguments:

            vec:      vector of variables, vec = [theta, mu]
            xn:       worker type/skill
            matching: one of either 'positive' or 'negative', depending.

        Returns:

            check: (boolean) True if montonicity condition satisfied.
            
        """
        # catch case where vec element is NaN
        if np.isnan(vec).any() == True:
            check = True
        
        else:
            LHS = (self.H(xn, vec) * 
                   self.derivatives_dict['Fyl'](xn, vec[1], vec[0], 1))

            if matching == 'positive':
                RHS = self.derivatives_dict['Fxr'](xn, vec[1], vec[0], 1)
            elif matching == 'negative':
                RHS = -self.derivatives_dict['Fxr'](xn, vec[1], vec[0], 1)
            else:
                raise ValueError
        
            # if True, then better firms hire more workers
            check = (LHS >= RHS)

        return check
        
    def set_solver(self, solver='OrthogonalCollocation', kind='Chebyshev'):
        """
        Sets the model's solver attribute.
        
        Arguments:
            
            solver: (str) A valid solver class. Must be one of 'Shooting',
                    'FiniteElements', or 'OrthogonalCollocation.' Default is
                    to use 'OrthogonalCollocation.'
            
            kind:   (str) If using an 'OrthogonalCollocation' solver, kind 
                    specifies the class of orthogonal polynomials to use as 
                    basis functions. Must be one of 'Chebyshev', 'Legendre', 
                    'Laguerre', or 'Hermite'.
                
        """
        if solver == 'OrthogonalCollocation':
            self.solver = OrthogonalCollocation(self, kind)
            
        elif solver == 'FiniteElements':
            self.solver = FiniteElements(self)
            
        elif solver == 'Shooting':
            self.solver = Shooting(self)
            
        else:
            raise ValueError
                
    ############### Methods for plotting ###############
    
    def plot_equilibrium_theta(self, ax=None, N=1000, xaxis='worker_skill', 
                               color='b'):
        """
        Generates a plot of the equilibrium firm size function, theta(x). User
        can plot firm size as a function of either worker skill, x, or firm
        productivity, y, by appropriately specifying the xaxis keyword arg.
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
                   
            N:      (int) Number of points to plot.
            
            xaxis:  (str) One of either 'worker_skill' or 'firm_productivity', 
                    depending. Default is 'worker_skill'.
                   
            color:  (various) Valid Matplotlib color.
                                
        """
        if ax == None:
            ax = plt.subplot(111)
            
        # create the grid of values against which to plot theta(x)
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N) 
        
        # extract the equilibrium theta(x)
        theta = self.equilibrium['theta']
        
        # plot the equilibrium theta as a function of...
        if xaxis == 'worker_skill': 
            ax.plot(x_grid, theta(x_grid), color=color),
            ax.set_xlabel('Worker skill, $x$', fontsize=15, family='serif')
            ax.set_ylabel(r'$\theta(x)$', rotation='horizontal', fontsize=15)
            
        elif xaxis == 'firm_productivity':
            
            # extract the matching function 
            mu = self.equilibrium['mu']
            
            # invert mu(x) = y using PCHIP interpolation 
            x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
            x_grid = np.linspace(x_lower, x_upper, N)
            
            # mu(x) may be increasing or decreasing!
            data = np.hstack((x_grid[:,np.newaxis], mu(x_grid)[:,np.newaxis]))
            data = data[data[:,1].argsort()]
            inverse_mu = interpolate.PchipInterpolator(data[:,1], data[:,0])
        
            # grid of values for plotting the matching function
            mu_lower = mu(x_lower)
            mu_upper = mu(x_upper)
            mu_grid  = np.linspace(mu_lower, mu_upper, N)
                                   
            ax.plot(mu_grid, theta(inverse_mu(mu_grid)), color=color),
            ax.set_xlabel('Firm productivity, $y$', fontsize=15, family='serif')
            ax.set_ylabel(r'$\theta(y)$', rotation='horizontal', fontsize=15)
                   
        else:
             raise ValueError ("'xaxis' must be one of 'worker_skill', or " +
                               "'firm_productivity'.")
    
    def plot_equilibrium_theta_dist(self, ax=None, N=1000, color='b', 
                                    which='cdf'):
        """
        Generates a plot of equilibrium distribution of firm size, theta.
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
                   
            N:      (int) Number of points to plot.
            
            color:  (various) Valid Matplotlib color.
            
            which:  (str) Which of 'pdf', 'cdf', or 'sf' do you want to plot?
                    Default is 'cdf'.
        
        """
        if ax == None:
            ax = plt.subplot(111)
                    
        # extract the equilibrium firm size
        theta = self.equilibrium['theta']
                 
        # invert theta(x) using PCHIP interpolation 
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N)
        
        # theta(x) may be increasing or decreasing!
        data = np.hstack((x_grid[:,np.newaxis], theta(x_grid)[:,np.newaxis]))
        data = data[data[:,1].argsort()]
        inverse_theta = interpolate.PchipInterpolator(data[:,1], data[:,0])
        
        # grid of values for plotting distribution of firm size
        theta_lower = theta(x_lower)
        theta_upper = theta(x_upper)
        theta_grid  = np.linspace(theta_lower, theta_upper, N)
            
        if which == 'pdf':
            pdf_size = (self.workers.scaled_pdf(inverse_theta(theta_grid)) * 
                         np.abs(inverse_theta.derivative(theta_grid, 1)))
            ax.plot(theta_grid, pdf_size, color=color)
            ax.set_xlabel(r'Firm size, $\theta$', fontsize=15, family='serif')
            ax.set_ylabel(r'Density (normalized)', fontsize=15, family='serif')
        
        elif which == 'cdf':
            
            # correct formula depends on whether inverse_theta is increasing!
            if (inverse_theta.derivative(theta_grid, 1) > -1e-3).all(): # KLUDGE!
                cdf_size = self.workers.scaled_cdf(inverse_theta(theta_grid))    
            else:
                cdf_size = self.workers.scaled_sf(inverse_theta(theta_grid))
            
            ax.plot(theta_grid, cdf_size, color=color)
            ax.set_xlabel(r'Firm size, $\theta$', fontsize=15, family='serif')
            ax.set_ylabel(r'Prob $(\Theta < \theta)$', fontsize=15, 
                          family='serif')
            
        elif which == 'sf':
            
            # correct formula depends on whether inverse_theta is increasing!
            if (inverse_theta.derivative(theta_grid, 1) > -1e-3).all():
                sf_size = self.workers.scaled_sf(inverse_theta(theta_grid))    
            else:
                sf_size = self.workers.scaled_cdf(inverse_theta(theta_grid))
                
            ax.plot(theta_grid, sf_size, color=color)
            ax.set_xlabel(r'Firm size, $\theta$', fontsize=15, family='serif')
            ax.set_ylabel(r'Prob $(\Theta > \theta)$', fontsize=15, 
                          family='serif')
                          
        else:
            raise ValueError, "'which' must be one of 'pdf', 'cdf', or 'sf'."
                     
    def plot_equilibrium_mu(self, ax=None, N=1000, color='b'):
        """
        Generates a plot of the equilibrium matching function, mu(x).
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
                   
            N:      (int) Number of points to plot.
            
            color:  (various) Valid Matplotlib color.
                    
        """
        if ax == None:
            ax = plt.subplot(111)
            
        # create the grid of values against which to plot theta(x)
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N) 
        
        # extract the equilibrium mu(x)
        mu = self.equilibrium['mu']
        
        # plot the equilibrium mu as a function of...
        ax.plot(x_grid, mu(x_grid), color=color),
        ax.set_xlabel('Worker skill, $x$', fontsize=15, family='serif')
        ax.set_ylabel(r'$\mu(x)$', rotation='horizontal', fontsize=15)    
                      
    def plot_equilibrium_wages(self, ax=None, N=1000, color='b'):
        """
        Generates a plot of the equilibrium wages, mu(x).
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
                   
            N:  (int) Number of points to plot.
                
        """
        if ax == None:
            ax = plt.subplot(111) 

        # create the grid of values against which to plot theta(x)
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N) 
        
        # extract the equilibrium mu(x)
        wages = self.equilibrium['wages']
        
        # plot the equilibrium wages as a function of...
        ax.plot(x_grid, wages(x_grid), color=color)
        ax.set_xlabel('Worker skill, $x$', fontsize=15, family='serif')
        ax.set_ylabel(r'$w(x)$', rotation='horizontal', fontsize=15)  
    
    def plot_equilibrium_wages_dist(self, ax=None, N=1000, color='b', 
                                    which='cdf'):
        """
        Generates a plot of equilibrium distribution of wages, w.
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
                   
            N:      (int) Number of points to plot.
            
            color:  (various) Valid Matplotlib color.
            
            which:  (str) Which of 'pdf', 'cdf', or 'sf' do you want to plot?
                               
        """
        if ax == None:
            ax = plt.subplot(111)
                    
        # extract the equilibrium wages
        w = self.equilibrium['wages']
                 
        # invert w(x) using PCHIP interpolation 
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N)
        inverse_w = interpolate.PchipInterpolator(w(x_grid), x_grid)
        
        # grid of values for plotting distribution of wages
        w_lower = w(x_lower)
        w_upper = w(x_upper)
        w_grid  = np.linspace(w_lower, w_upper, N)
            
        if which == 'pdf':
            pdf_wages = (self.workers.scaled_pdf(inverse_w(w_grid)) * 
                         np.abs(inverse_w.derivative(w_grid, 1)))
            ax.plot(w_grid, pdf_wages, color=color)
            ax.set_xlabel('Real wage, $w$', fontsize=15, family='serif')
            ax.set_ylabel(r'Density (normalized)', fontsize=15, family='serif')
        
        elif which == 'cdf':
            cdf_wages = self.workers.scaled_cdf(inverse_w(w_grid))
            ax.plot(w_grid, cdf_wages, color=color)
            ax.set_xlabel('Real wage, $w$', fontsize=15, family='serif')
            ax.set_ylabel(r'Prob $(W < w)$', fontsize=15, family='serif')
            
        elif which == 'sf':
            sf_wages = self.workers.scaled_sf(inverse_w(w_grid))
            ax.plot(w_grid, sf_wages, color=color)
            ax.set_xlabel('Real wage, $w$', fontsize=15, family='serif')
            ax.set_ylabel(r'Prob $(W > w)$', fontsize=15, family='serif')
                          
        else:
            raise ValueError, "'which' must be one of 'pdf', 'cdf,' or 'sf'."
                        
    def plot_equilibrium_profits(self, ax=None, N=1000, xaxis='worker_skill', 
                                 color='b'):
        """
        Generates a plot of the equilibrium firm size function, theta(x).
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
            
            N:      (int) Number of points to plot.
            
            xaxis:  (str) One of either 'worker_skill' or 'firm_productivity', 
                    depending. Default is 'worker_skill'.
                   
            color:  (various) Valid Matplotlib color.
        
        """
        if ax == None:
            ax = plt.subplot(111) 

        # create the grid of values against which to plot theta(x)
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N) 
        
        # extract the equilibrium pi(x)
        profits = self.equilibrium['profits']
        
        # plot the equilibrium theta as a function of...
        if xaxis == 'worker_skill': 
            ax.plot(x_grid, profits(x_grid), color=color)
            ax.set_xlabel('Worker skill, $x$', fontsize=15, family='serif')
            ax.set_ylabel(r'$\pi(x)$', rotation='horizontal', fontsize=15)
            
        elif xaxis == 'firm_productivity':
            # extract the matching function 
            mu = self.equilibrium['mu']
            
            # invert mu(x) = y using PCHIP interpolation 
            x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
            x_grid = np.linspace(x_lower, x_upper, N)
            
            # mu(x) may be increasing or decreasing!
            data = np.hstack((x_grid[:,np.newaxis], mu(x_grid)[:,np.newaxis]))
            data = data[data[:,1].argsort()]
            inverse_mu = interpolate.PchipInterpolator(data[:,1], data[:,0])
                    
            # grid of values for plotting the matching function
            mu_lower = mu(x_lower)
            mu_upper = mu(x_upper)
            mu_grid  = np.linspace(mu_lower, mu_upper, N)
                                   
            ax.plot(mu_grid, profits(inverse_mu(mu_grid)), color=color)
            ax.set_xlabel('Firm productivity, $y$', fontsize=15, family='serif')
            ax.set_ylabel(r'$\pi(y)$', rotation='horizontal', fontsize=15)
                   
        else:
             raise ValueError, ("'xaxis' must be one of 'worker_skill' or " + 
                                "'firm_productivity'.")
             
    def plot_equilibrium_profits_dist(self, ax=None, N=1000, color='b', 
                                      which='cdf'):
        """
        Generates a plot of equilibrium distribution of profits, pi.
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
                   
            N:      (int) Number of points to plot.
            
            color:  (various) Valid Matplotlib color.
            
            which:  (str) Which of 'pdf', 'cdf', or 'sf' do you want to plot?
                    Default is 'cdf'.
        
        """
        if ax == None:
            ax = plt.subplot(111)
                    
        # extract the equilibrium profits
        pi = self.equilibrium['profits']
                 
        # invert pi(x) using PCHIP interpolation
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N)
        inverse_pi = interpolate.PchipInterpolator(pi(x_grid), x_grid)
        
        # grid of values for plotting distribution of profits
        pi_lower = pi(x_lower)
        pi_upper = pi(x_upper)
        pi_grid  = np.linspace(pi_lower, pi_upper, N)
            
        if which == 'pdf':
            pdf_profits = (self.workers.scaled_pdf(inverse_pi(pi_grid)) * 
                           np.abs(inverse_pi.derivative(pi_grid, 1)))
            ax.plot(pi_grid, pdf_profits, color=color)
            ax.set_xlabel(r'Firm profits, $\pi$', fontsize=15, family='serif')
            ax.set_ylabel(r'Density (normalized)', fontsize=15, family='serif')
        
        elif which == 'cdf':
            cdf_profits = self.workers.scaled_cdf(inverse_pi(pi_grid))
            ax.plot(pi_grid, cdf_profits, color=color)
            ax.set_xlabel('Firm profits, $\pi$', fontsize=15, family='serif')
            ax.set_ylabel(r'Prob $(\Pi < \pi)$', fontsize=15, family='serif')
            
        elif which == 'sf':
            sf_profits = self.workers.scaled_sf(inverse_pi(pi_grid))
            ax.plot(pi_grid, sf_profits, color=color)
            ax.set_xlabel('Firm profits, $\pi$', fontsize=15, family='serif')
            ax.set_ylabel(r'Prob $(\Pi > \pi)$', fontsize=15, family='serif')
                          
        else:
            raise ValueError, "'which' must be one of 'pdf', 'cdf,' or 'sf'."
                    
    def plot_equilibrium_funcs(self, fig=None, axes=None, N=1000, 
                               xaxis='worker_skill', color='b'):
        """
        Generates a plot of the equilibrium functions theta(x), mu(x), w(x), and
        pi(x). Equilibrium firm size, theta, and profits, pi, can be plotted as
        functions of firm productivity, y, by appropriately specifying the 
        keyword argument xaxis.
        
        Arguments:
            
            fig:    (object) Figure object.
            
            axes:   (list) List of AxesSubplot objects.
            
            N:      (int) Number of points to plot.
            
            xaxis:  (str) One of either 'worker_skill' or 'firm_productivity', 
                    depending. Default is 'worker_skill'.
                   
            color:  (various) Valid Matplotlib color.
                    
        """
        if fig == None and axes == None:
            fig, axes = plt.subplots(2, 2, figsize=(8,8))
        
        # plot the equilibrium functions
        self.plot_equilibrium_theta(axes[0,0], N, xaxis, color)
        self.plot_equilibrium_mu(axes[0,1], N, color)
        self.plot_equilibrium_profits(axes[1,0], N, xaxis, color)
        self.plot_equilibrium_wages(axes[1,1], N, color)
        
        fig.tight_layout()

    def plot_equilibrium_output_per_worker(self, ax=None, N=1000,
                                           xaxis='worker_skill', color='b'):
        """
        Generates a plot of equilibrium output per worker. User can plot output  
        per worker as a unction of either worker skill, x, or firm productivity,
        y, by appropriately specifying the xaxis keyword arg.
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
                   
            N:      (int) Number of points to plot.
            
            xaxis:  (str) One of either 'worker_skill' or 'firm_productivity', 
                    depending. Default is 'worker_skill'.
                   
            color:  (various) Valid Matplotlib color.
                    
        """
        if ax == None:
            ax = plt.subplot(111)
            
        # create the grid of values against which to plot theta(x)
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N) 
        
        # extract the equilibrium theta(x) and mu(x)
        theta = self.equilibrium['theta']
        mu = self.equilibrium['mu']
        
        # plot the equilibrium output as a function of...
        if xaxis == 'worker_skill': 
            output_per_worker = self.output(x_grid, mu(x_grid), theta(x_grid), 1) / theta(x_grid)
            ax.plot(x_grid, output_per_worker, color=color)
            ax.set_xlabel('Worker skill, $x$', fontsize=15, family='serif')
            ax.set_ylabel(r'Output per worker', rotation='horizontal', 
                          fontsize=15, family='serif')
            
        elif xaxis == 'firm_productivity':
            
            # invert mu(x) = y using PCHIP interpolation 
            x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
            x_grid = np.linspace(x_lower, x_upper, N)
            
            # mu(x) may be increasing or decreasing!
            data = np.hstack((x_grid[:,np.newaxis], mu(x_grid)[:,np.newaxis]))
            data = data[data[:,1].argsort()]
            inverse_mu = interpolate.PchipInterpolator(data[:,1], data[:,0])
        
            # grid of values for plotting the matching function
            mu_lower = mu(x_lower)
            mu_upper = mu(x_upper)
            mu_grid  = np.linspace(mu_lower, mu_upper, N)
            
            output = self.output(inverse_mu(mu_grid), mu_grid, 
                                 theta(inverse_mu(mu_grid)), 1)
            output_per_worker = output / theta(inverse_mu(mu_grid))
            
            ax.plot(mu_grid, output_per_worker,color=color)
            ax.set_xlabel('Firm productivity, $y$', fontsize=15, family='serif')
            ax.set_ylabel(r'Output per worker', rotation='horizontal', 
                          fontsize=15, family='serif')
                   
        else:
             raise ValueError, ("'xaxis' must be one of 'worker_skill' or " + 
                                "'firm_productivity'.")
                                
    def plot_equilibrium_output_per_worker_dist(self, ax=None, N=1000, 
                                                color='b', which='cdf'):
        """
        Generates a plot of equilibrium distribution of output per worker, f.
        
        Arguments:
            
            ax:     (object) AxesSubplot instance on which to place the plot.
                   
            N:      (int) Number of points to plot.
            
            color:  (various) Valid Matplotlib color.
            
            which:  (str) Which of 'pdf', 'cdf', or 'sf' do you want to plot?
                    Default is 'cdf'.
                    
        """
        if ax == None:
            ax = plt.subplot(111)
                    
        # extract the equilibrium theta(x) and mu(x)
        theta = self.equilibrium['theta']
        mu    = self.equilibrium['mu']
                 
        # invert F(x) using PCHIP interpolation
        x_lower, x_upper = self.workers.lower_bound, self.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N)
        f = self.output(x_grid, mu(x_grid), theta(x_grid), 1) / theta(x_grid)
        inverse_f = interpolate.PchipInterpolator(f, x_grid)
        
        # grid of values for plotting distribution of output
        f_lower = f.min()
        f_upper = f.max()
        f_grid  = np.linspace(f_lower, f_upper, N)
            
        if which == 'pdf':
            pdf_f = (self.workers.scaled_pdf(inverse_f(f_grid)) * 
                     np.abs(inverse_f.derivative(f_grid, 1)))
            ax.plot(f_grid, pdf_f, color=color)
            ax.set_xlabel('Firm output per worker, $f$', fontsize=15, family='serif')
            ax.set_ylabel(r'Density (normalized)', fontsize=15, family='serif')
        
        elif which == 'cdf':
            cdf_f = self.workers.scaled_cdf(inverse_f(f_grid)) 
            ax.plot(f_grid, cdf_f, color=color)
            ax.set_xlabel('Firm output per worker, $f$', fontsize=15, family='serif')
            ax.set_ylabel(r'Prob $(f < x)$', fontsize=15, family='serif')
            
        elif which == 'sf':
            sf_F = self.workers.scaled_sf(inverse_f(f_grid)) 
            ax.plot(f_grid, sf_F, color=color)
            ax.set_xlabel('Firm output per worker, $f$', fontsize=15, family='serif')
            ax.set_ylabel(r'Prob $(f > x)$', fontsize=15, family='serif')
                          
        else:
            raise ValueError, "'which' must be one of 'pdf', 'cdf,' or 'sf'."

class Shooting(object):
    """Base class representing a simple shooting solver."""
    
    def __init__(self, model):
         """
         Initializes an instance of the OrthogonalCollocation class with the 
         following attributes:
            
             model: (object) An instance of the Model class.
                        
         """
         self.model = model
         
         # shooting method computes the equilibrium path
         self.success = True
         self.path = None
         
    def bspline(self, knots, values, deg=1, der=0, ext=0):
        """
        Approximates a function using a B-spline.
        
        Arguments:
                        
            knots:  (array) (N, ) array of B-spline knots.
            
            values: (array) (N + 1,) array of B-spline coefficients.
            
            deg:    (int) Degree of desired B-spline. Must satisfy 1 <= deg <= 5.
            
            der:    (int) Derivative of the B-spline. Default is zero.
            
            ext:    (int) How to handle extrapolation. Default is zero. See the
                    docstring for interpolate.splev for more details.
            
        Returns:
            
            bspline: (callable) B-spline representation of a function.
            
        """
        tck       = interpolate.splrep(knots, values, k=deg, s=0) 
        bspline   = lambda x: interpolate.splev(x, tck, der, ext)
        
        return bspline
       
    def residual(self, x, deg, matching='pam'):
        """
        Computes the residual for the Shooting method.
        
        Arguments:

            x:        (array) Worker type/skill.
            
            deg:    (int) Degree of desired B-spline. Must satisfy 1 <= deg <= 5.
                 
            matching: (str) Whether desired matching equilibrium should exhibit 
                      postive ('pam') or negative ('nam') assortative matching.
                      Default is 'pam'.
                    
        """
        # construct the B-spline approximations of mu(x) and theta(x)
        theta   = self.bspline(self.path[:,0], self.path[:,1], deg, 0)
        theta_x = self.bspline(self.path[:,0], self.path[:,1], deg, 1)
        
        mu   =  self.bspline(self.path[:,0], self.path[:,2], deg, 0)
        mu_x =  self.bspline(self.path[:,0], self.path[:,2], deg, 1)
        
        # compute the residuals 
        LHS       = theta_x(x)
        RHS       = self.model.theta_prime(x, [theta(x), mu(x)], matching)
        res_theta = LHS - RHS
          
        LHS    = mu_x(x)
        RHS    = self.model.mu_prime(x, [theta(x), mu(x)], matching)
        res_mu = LHS - RHS
                     
        res    = np.hstack((res_theta, res_mu)) 
        
        return res
    
    def system(self, x, vec, matching='pam'):
        """
        2D system of differential equations describing the behavior of an 
        equilibrium with positive assortative matching between workers and 
        firms.
    
        Arguments:

            x:        (array) Worker type/skill.
            
            vec:      (array) Vector of endogenous variables: factor intensity, 
                      theta(x), and matching assignment, mu(x). Ordering of the 
                      variables is [theta(x), mu(x)].
                 
            matching: (str) Whether desired matching equilibrium should exhibit 
                      postive ('pam') or negative ('nam') assortative matching.
                      Default is 'pam'.
                    
        Returns:
                
            out: (array) Vector of values [theta'(x), mu'(x)].
    
        """
        out = np.hstack((self.model.theta_prime(x, vec, matching),
                         self.model.mu_prime(x, vec, matching)))
            
        return out
        
    def solve(self, theta_0, tol=1e-2, N=1e3, deg=1, matching='pam',
              mesg=False, max_iter=1e6, integrator='lsoda', **kwargs):
        """
        Solve for the equilibrium of the Eeckhout and Kircher (2013) model 
        using a Forward Shooting algorithm. For details on the forward shooting 
        algorithm see Judd (1992) p. 357.

        Arguments:

        theta_0:    Initial guess for the size of the most productive firm.
        
        tol:        How close to the shooting target is "close enough."
        
        N:          Number of points of the solution path to compute.
        
        deg:        Degree of B-spline approximation to use. Must satisfy
                    1 <= deg <= 5.
                    
        matching:   One of either 'pam' or 'nam' depending on whether or not you
                    want an equilibrium with positive or negative assortative 
                    matching.
        
        mesg:       Do you want to print messages indicating progress
                    towards convergence? Default is False.
        
        integrator: Solution method for ODE solver.  Must be one of 'vode', 
                    'zvode', 'dopri5', 'dop853'.  See scipy.integrate.ode for 
                    details (including references for algorithms).
        
        **kwargs:   Additional arguments to pass to the ODE solver.  
                    See scipy.integrate.ode for details.
        
        """
        # which or either positive or negative assortative matching?
        if matching == 'pam':
            self.__shoot_pam(theta_0, tol, N, deg, mesg, max_iter, integrator, 
                             **kwargs)

        elif matching == 'nam':
            self.__shoot_nam(theta_0, tol, N, deg, mesg, max_iter, integrator, 
                             **kwargs)
        
        else:
            raise ValueError
                    
    def __shoot_pam(self, theta_0, tol, N, deg, mesg, max_iter, integrator, 
                    **kwargs):
        """Uses a forward shooting algorithm to solve for a PAM euilibrium."""

        # range of worker types 
        xl = self.model.workers.lower_bound
        xu = self.model.workers.upper_bound
            
        # range of worker types 
        yl = self.model.firms.lower_bound 
        yu = self.model.firms.upper_bound
    
        # initial condition for the solver 
        init = np.array([theta_0, yu])

        # compute the optimal step size
        step_size = (xu - xl) / (N - 1)
            
        # check that initial wages and profits are strictly positive
        if self.model.get_wages(xu, init) <= 0.0:
            raise Exception, ('Invalid initial condition! Most skilled worker' +
                              ' must earn strictly positive wage!')

        if self.model.get_profits(xu, init) <= 0.0:
            raise Exception, ('Invalid initial condition! Most productive' +
                              ' firm must earn strictly positive profits!')

        # check that PAM holds for initial condition
        if not self.model.check_pam(xu, init):
            raise Exception, ('Invalid initial condition! Necessary condition' +
                              ' for PAM fails!')
            
        ########## ODE solver ##########
    
        # initialize solver for PAM
        solver = integrate.ode(self.system)
        solver.set_f_params('pam')
        solver.set_integrator(integrator, **kwargs)
        solver.set_initial_value(init, xu)
                
        ##### Forward shooting algorithm #####
    
        # initialize putative equilibrium path
        init_wages   = self.model.get_wages(xu, init)
        init_profits = self.model.get_profits(xu, init)
        path         = np.hstack((xu, init, init_wages, init_profits)) 

        # initialize a counter
        num_iter = 0

        # initial feasible range for theta_0 
        theta_h = 2.0 * theta_0
        theta_l = 0
        
        while num_iter < max_iter:
            num_iter += 1
            if mesg == True and num_iter % 1000 == 0:
                print 'Completed', num_iter, 'iterations.' 
            
            # Walk the 2D system forward one step
            solver.integrate(solver.t - step_size)
        
            # compute profits and wages along putative equilibrium
            tmp_wages   = self.model.get_wages(solver.t, solver.y)
            tmp_profits = self.model.get_profits(solver.t, solver.y)
            tmp_step    = np.hstack((solver.t, solver.y, tmp_wages, tmp_profits))
            path        = np.vstack((path, tmp_step))

            ##### At each step, need to... #####

            # check that necessary condition for PAM holds
            if not self.model.check_pam(solver.t, solver.y):
                print 'Necessary condition for PAM failed!'
                print 'Most recent step:', path[-1,:]
                break

            # check that firm size is non-negative!
            if solver.y[0] <= tol:

                # theta_0 too low  
                theta_l = init[0]
                init[0] = (theta_h + theta_l) / 2
                if mesg == True:
                    print 'Negative firm size! Guess for theta_0 was too low!'
                    print 'Most recent step:', path[-1,:]
                    print 'New initial condition is', init[0]
                    print ''
                solver.set_initial_value(init, xu)
                
                # reset the putative equilibrium path
                init_wages = self.model.get_wages(xu, init)
                init_profits = self.model.get_profits(xu, init)
                path = np.hstack((xu, init, init_wages, init_profits))

            # check workers...
            elif solver.t <= xl:
                
                # ...check for equilibrium with all workers/firms matched! 
                if (np.abs(solver.t - xl) < tol and 
                    np.abs(solver.y[1] - yl) < tol):
                    if mesg == True:
                        print 'Sucess! All workers matched with all firms.'
                        print 'Most recent step:', path[-1,:]
                    break

                # ...check for equilibrium with excess supply of firms!
                elif np.abs(solver.t - xl) < tol and np.abs(tmp_profits) < tol:
                    if mesg == True:
                        print 'Found equilibrium with excess supply of firms!'
                        print 'Most recent step:', path[-1,:]
                    break

                # ...check if exhausted all firms (i.e., theta_0 too low!)
                elif (solver.y[1] - yl) < -tol:
                    theta_l = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print ('You have run out of firms! Guess for theta_0' + 
                               ' was too low!')
                        print 'Most recent step:', path[-1,:]
                        print 'New initial condition is', init[0]
                        print ''
                    solver.set_initial_value(init, xu)
                
                    # reset the putative equilibrium path
                    init_wages = self.model.get_wages(xu, init)
                    init_profits = self.model.get_profits(xu, init)
                    path = np.hstack((xu, init, init_wages, init_profits))
                
                # ...else, theta_0 too high!
                else:
                    theta_h = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print('You have run out of workers, but there are ' + 
                              'still firms around and profits are non-zero.' +
                              ' Guess for theta_0 was too high!') 
                        print 'Most recent step:', path[-1,:]
                        print 'New initial condition is', init[0]
                        print ''
                    solver.set_initial_value(init, xu)
                
                    # reset the putative equilibrium path
                    init_wages = self.model.get_wages(xu, init)
                    init_profits = self.model.get_profits(xu, init)
                    path = np.hstack((xu, init, init_wages, init_profits))
                        
            # check firms...
            elif solver.y[1] <= yl:
                
                # ...check for equilibrium with all workers/firms matched!
                if (np.abs(solver.t - xl) < tol and 
                    np.abs(solver.y[1] - yl) < tol):
                    if mesg == True:
                        print 'Sucess! All workers matched with all firms.'
                        print 'Most recent step:', path[-1,:]
                    break

                # ...check for equilibrium with excess supply of workers!
                elif (np.abs(tmp_wages) < tol and 
                        np.abs(solver.y[1] - yl) < tol):
                    if mesg == True:
                        print 'Found equilibrium with excess supply of workers!'
                        print 'Most recent step:', path[-1,:]
                    break

                # ...check if exhausted all workers (i.e., theta_0 too high!)
                elif (solver.t - xl) < -tol:
                    theta_h = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print ('You have run out of workers! Guess for' +
                               ' theta_0 was too high!') 
                        print 'Most recent step:', path[-1,:]
                        print 'New initial condition is', init[0]
                        print ''
                    solver.set_initial_value(init, xu)
                
                    # reset the putative equilibrium path
                    init_wages = self.model.get_wages(xu, init)
                    init_profits = self.model.get_profits(xu, init)
                    path = np.hstack((xu, init, init_wages, init_profits))
                
                # ...else, theta_0 too low!
                else:
                    theta_l = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print('You have run out of firms, but there are ' + 
                              'still workers around and wages are non-zero.' +
                              ' Guess for theta_0 was too low!')
                        print 'Most recent step:', path[-1,:]
                        print 'New initial condition is', init[0]
                        print ''
                    solver.set_initial_value(init, xu)
                
                    # reset the putative equilibrium path
                    init_wages = self.model.get_wages(xu, init)
                    init_profits = self.model.get_profits(xu, init)
                    path = np.hstack((xu, init, init_wages, init_profits))
            
            # check that wages are non-negative...
            elif tmp_wages <= 0:  

                # check if exhausted all firms and wages are zeroish...
                if np.abs(tmp_wages) < tol and np.abs(solver.y[1] - yl) < tol:
                    if mesg == True:
                        print('Found equilibrium with excess supply of workers' + 
                              '...but, you should not be reading this!')
                        print 'Most recent step:', path[-1,:]
                    break

                # theta_0 too high!  
                else:
                    theta_h = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print('There are still workers and firms around, ' + 
                              'but wages are zero! Guess for theta_0 was too' +
                              'high!')
                        print 'Most recent step:', path[-1,:]
                        print 'New initial condition is', init[0]
                        print ''
                    solver.set_initial_value(init, xu)
                
                    # reset the putative equilibrium path
                    init_wages = self.model.get_wages(xu, init)
                    init_profits = self.model.get_profits(xu, init)
                    path = np.hstack((xu, init, init_wages, init_profits))

            # check that profits are non-negative...
            elif tmp_profits <= 0:

                # check if exhausted all workers and profits are zeroish
                if np.abs(tmp_profits) < tol and np.abs(solver.t - xl) < tol:
                    if mesg == True:
                        print('Found equilibrium with excess supply of firms' + 
                              '...but, you should not be reading this!')
                    break

                # theta_0 too low!
                else:
                    theta_l = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print('There are still workers and firms around, ' + 
                              'but profits are zero!')
                        print 'Most recent step:', path[-1,:]
                        print 'New initial condition is', init[0]
                        print ''
                    solver.set_initial_value(init, xu)
                
                    # reset the putative equilibrium path
                    init_wages = self.model.get_wages(xu, init)
                    init_profits = self.model.get_profits(xu, init)
                    path = np.hstack((xu, init, init_wages, init_profits))
                    
            # if all of the above are satisfied, then continue!
            else:
                continue     
            
            # check whether max_iter condition has been reached
            if num_iter == int(max_iter):
                print "Reached maximum iterations w/o finding a solution."
                self.success = False
                break
                  
        # modify the solver's path attribute (reverse order!)
        self.path = path[::-1]  
        
        if self.success == True:
            
            # approximate equilibrium firm size and matching functions
            theta = self.bspline(self.path[:,0], self.path[:,1], deg)
            mu    = self.bspline(self.path[:,0], self.path[:,2], deg)
            
            # approximate equilibrium wages and profits
            w  = lambda x: self.model.get_wages(x, [theta(x), mu(x)])
            pi = lambda x: self.model.get_profits(x, [theta(x), mu(x)])
            
            # modify the equilibrium attribute
            self.model.equilibrium['theta']   = theta
            self.model.equilibrium['mu']      = mu
            self.model.equilibrium['wages']   = w
            self.model.equilibrium['profits'] = pi 

    def __shoot_nam(self, system, theta_0, tol, N, deg, mesg, max_iter, 
                    integrator, **kwargs):
        raise Exception, NotImplementedError
     
    def plot_residual(self, N=1000, matching='pam'):
        """
        Generates a plot of the shooting residuals for theta(x) and mu(x).
        
        Arguments:
            
            N: (int) Number of points to plot.
            
        """    
        ax0 = plt.subplot(211) 
        ax1 = plt.subplot(212) 

        x_lower = self.model.workers.lower_bound
        x_upper = self.model.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, N) 
        
        # define the residual function
        resid = self.residual(x_grid, 1, matching)        
        
        # plot the theta residual
        ax0.plot(x_grid, resid[:N])
        ax0.set_ylabel(r'$R_{\theta}$', rotation='horizontal', fontsize=15)
        ax0.set_title(r'Shooting residuals for $\theta(x)$ and $\mu(x)$.',
                      fontsize=15, family='serif')
                      
        # plot the mu residual
        ax1.plot(x_grid, resid[N:])
        ax1.set_xlabel('$x$', fontsize=15)
        ax1.set_ylabel(r'$R_{\mu}$', rotation='horizontal', fontsize=15)

class OrthogonalCollocation(object):
    """Base class representing a simple orthogonal collocation solver."""
    
    def __init__(self, model, kind='Chebyshev'):
        """
        Initializes an instance of the OrthogonalCollocation class with the 
        following attributes:
            
            model: (object) An instance of the Model class.
            
            kind:  (str) The class of orthogonal polynomials to use as basis 
                   functions. Default is Chebyshev polynomials.
            
        """
        self.model = model
        self.kind  = kind
        
        # model equilibrium is completely described by the following
        self.nodes   = None
        self.coefs   = None
        self.x_lower = None
        self.x_upper = None
        self.y_lower = None
        self.y_upper = None
        
    def polynomial(self, coefs):
        """
        Orthogonal polynomial approximation of theta(x).
        
        Arguments:
            
            coefs:   (array-like) Chebyshev polynomial coefficients.
            
        Returns:
            
            poly: (object) Instance of an orthogonal polynomial class.
            
        """        
        domain = [self.x_lower, self.x_upper]
        
        if self.kind == 'Chebyshev':
            poly = Chebyshev(coefs, domain)
        elif self.kind == 'Legendre':
            poly = Legendre(coefs, domain)
        elif self.kind == 'Laguerre':
            poly = Laguerre(coefs)
        elif self.kind == 'Hermite':
            poly = Hermite(coefs)
        else:
            raise ValueError

        return poly

    def residual(self, x, coefs, matching):
        """
        Residual function for orthogonal collocation solution methods.
        
        Arguments:
        
            x:        (array-like) Value(s) of worker skill.
            
            coefs:    (array-like) (2, N) array of orthogonal polynomial 
                      coefficients. The first row of coefficients is used to 
                      constuct the theta_hat(x) approximation of the equilibrium 
                      firm size function theta(x). The second row of
                      coefficients is used to construct the mu_hat(x) 
                      approximation of the equilibrium matching function mu(x). 
                
            matching: (str) One of either 'pam' or 'nam', depending on whether
                      or not you wish to compute a positive or negative 
                      assortative equilibrium.
                      
        Returns:
            
            residual: (array) Collocation residuals.    
                            
        """        
        # construct the polynomial approximations of mu(x) and theta(x)
        theta   = self.polynomial(coefs[0])
        theta_x = theta.deriv()
        
        mu      = self.polynomial(coefs[1])
        mu_x    = mu.deriv()
        
        # compute the residual polynomial
        res_mu    = (mu_x(x) - 
                     self.model.mu_prime(x, [theta(x), mu(x)], matching))
        res_theta = (theta_x(x) - 
                     self.model.theta_prime(x, [theta(x), mu(x)], matching))
            
        residual  = np.hstack((res_theta, res_mu)) 
        
        return residual
    
    def system(self, coefs, nodes, matching):
        """
        System of non-linear equations for orthogonal collocation method.
        
        Arguments:
            
            coefs:    (array) (2, m) array of orthogonal polynomial coefficients. 
                      The first row of coefficients is used to constuct the 
                      theta_hat(x) approximation of the equilibrium firm size 
                      function theta(x). The second row of coefficients is used
                      to construct the mu_hat(x) approximation of the 
                      equilibrium matching function mu(x). 
                
            nodes:    (array) (m,) array of polynomial roots used as collocation 
                      nodes. The number of collocation nodes plus the number of 
                      boundary conditions must equal the number of unknown 
                      coefficients.
                                      
            matching: (str) One of either 'pam' or 'nam', depending on whether
                      or not you wish to compute a positive or negative 
                      assortative equilibrium.
                   
        Returns:
            
            out: (array) Array of residuals.
            
        """
        deg     = nodes.size
        coefs   = coefs.reshape((2, deg + 1))
        
        # compute the equilibrium matching function at the boundary-points
        mu = self.polynomial(coefs[1])
        boundary_conditions = np.array([mu(self.x_lower) - self.y_lower, 
                                        mu(self.x_upper) - self.y_upper])
        
        # compute the value of the residual for given set of coefs
        out = np.hstack((boundary_conditions, 
                         self.residual(nodes, coefs, matching)))
        
        return out

    def solve(self, init_coefs, bc, matching='pam', method='hybr', **kwargs):
        """
        Solves for the equilibrium functions theta(x) and mu(x) using Chebyshev 
        collocation.
        
        Arguments:
                        
            init_coefs: (array) (2, deg + 1) array of coefficients representing 
                        a guess of the optimal set of coefficients.
                        
            bc:         (list) List of tuples defining the boundary conditions. 
                        Tuples should be (x, mu(x)) pairs which pin down the 
                        value of equilibrium matching function at the end-points.
            
            matching:   (str) One of either 'pam' or 'nam', depending on whether
                        or not you wish to compute a positive or negative 
                        assortative equilibrium.
            
            method:     (str) Method used to solve system of non-linear 
                        equations.
              
        If the boundary value problem was successfully solved, then the 
        equilibrium functions theta(x), mu(x), w(x) and pi(x) are stored in the
        model's equilibrium dictionary. If a solution was not found, then the 
        result object is returned for inspection.
                                  
        """
        # lower boundary condition
        self.x_lower = bc[0][0]
        self.y_lower = bc[0][1]

        # upper boundary condition
        self.x_upper = bc[1][0]
        self.y_upper = bc[1][1]
        
        # get additional collocation nodes (should I always use Chebyshev nodes?)
        basis_coefs     = np.zeros(init_coefs.shape[1])
        basis_coefs[-1] = 1
        nodes           = self.polynomial(basis_coefs).roots()
        
        # solve the system of non-linear equations 
        res = optimize.root(self.system, init_coefs.flatten(), 
                            args=(nodes, matching), method=method, **kwargs)
        
        # if solution has been found, modify the equilibrium attribute 
        if res.success == True:
            print res.message
            
            # extract the solution coefficients 
            self.coefs = res.x.reshape(init_coefs.shape)
            
            # approximate equilibrium firm size and matching functions
            theta   = self.polynomial(self.coefs[0])
            mu      = self.polynomial(self.coefs[1])  
            
            # approximate equilibrium wages and profits
            w  = lambda x: self.model.get_wages(x, [theta(x), mu(x)])
            pi = lambda x: self.model.get_profits(x, [theta(x), mu(x)])
            
            # modify the equilibrium attribute
            self.model.equilibrium['theta']   = theta
            self.model.equilibrium['mu']      = mu
            self.model.equilibrium['wages']   = w
            self.model.equilibrium['profits'] = pi 
            
        else:
            print res.message
            
        return res

    def plot_residual(self, N=1000, matching='pam'):
        """
        Generates a plot of the collocation residuals for theta(x) and mu(x).
        
        Arguments:
            
            N:        (int) Number of points to plot.
            
            matching: (str) One of either 'pam' or 'nam', depending on whether
                      or not you wish to compute a positive or negative 
                      assortative equilibrium.
            
        """    
        ax0 = plt.subplot(211) 
        ax1 = plt.subplot(212) 

        # compute the residual at some grid of x values
        x_grid = np.linspace(self.x_lower, self.x_upper, N) 
        resid = self.residual(x_grid, self.coefs, matching)

        # plot the theta residual
        ax0.plot(x_grid, resid[:N])
        #ax0.set_yscale('log')
        ax0.set_ylabel(r'$R_{\theta}$', rotation='horizontal', fontsize=15)
        ax0.set_title(r'Collocation residuals for $\theta(x)$ and $\mu(x)$.',
                      fontsize=15, family='serif')
                      
        # plot the mu residual
        ax1.plot(x_grid, resid[N:])
        #ax1.set_yscale('log')
        ax1.set_xlabel('$x$', fontsize=15)
        ax1.set_ylabel(r'$R_{\mu}$', rotation='horizontal', fontsize=15)
        
class FiniteElements(object):
    """Base class representing a simple finite-elements solver."""
    
    def __init__(self, model):
        """
        Initializes an instance of the FiniteElements class with the following
        attributes:
            
            model: (object) An instance of the Model class.
            
        """
        self.model = model
        
        # solution is completely described by the following
        self.coefs = None
        self.knots = None
        self.deg   = None
        
    def bspline(self, x, knots, coefs, deg, der=0, ext=0):
        """
        Approximates a function using a B-spline.
        
        Arguments:
            
            x:     (array) Points at which to evaluate the B-spline.
            
            knots: (array) (N, ) array of B-spline knots.
            
            coefs: (array) (N + 1,) array of B-spline coefficients.
            
            deg:   (int) Degree of desired B-spline. Must satisfy 1 <= deg <= 5.
            
            der:   (int) Derivative of the B-spline. Default is zero.
            
            ext:   (int) How to handle extrapolation. Default is zero. See the
                   docstring for interpolate.splev for more details.
            
        Returns:
            
            bspline: (array) B-spline representation of a function.
            
        """
        knots = np.hstack((knots[0], knots, knots[-1]))
        coefs = np.hstack((coefs, np.zeros(2)))
        
        bspline = interpolate.splev(x, (knots, coefs, deg), der, ext)
        return bspline
        
    def residual(self, x, knots, coefs, deg, matching):
        """
        System of non-linear equations for collocation method.
        
        Arguments:
            
            x:     (array) Value(s) of worker skill.
            
            knots: (array) (2, N) array of B-spline knots. The first row of 
                   knots is used to constuct the B-spline approximation of the 
                   equilibrium firm size function theta(x). The second row of 
                   knots is used to construct the B-spline approximation of the 
                   equilibrium matching function mu(x). 
                
            coefs: (array) (2, N + 1) array of B-spline coefficients. The first 
                   row of coefficients is used to constuct the B-spline 
                   approximation of the equilibrium firm size function theta(x).
                   The second row of coefficients is used to construct the 
                   B-spline approximation of the equilibrium matching function 
                   mu(x). 
                   
            deg:   (int) Degree of desired B-spline. Must satisfy 1 <= deg <= 5.
                   
        Returns:
            
            out: (array) Array of residuals.
            
        """
        # construct the B-spline approximations of mu(x) and theta(x)
        theta   = lambda x: self.bspline(x, knots[0], coefs[0], deg, 0)
        theta_x = lambda x: self.bspline(x, knots[0], coefs[0], deg, 1)
        
        mu   =  lambda x: self.bspline(x, knots[1], coefs[1], deg, 0)
        mu_x =  lambda x: self.bspline(x, knots[1], coefs[1], deg, 1)
        
        # compute the residual 
        res_theta = (theta_x(x[0]) - 
                     self.model.theta_prime(x[0], [theta(x[0]), mu(x[0])],  matching))
        res_mu    = (mu_x(x[1]) - 
                     self.model.mu_prime(x[1], [theta(x[0]), mu(x[0])], matching))
        res       = np.hstack((res_theta, res_mu)) 
        
        return res
        
    def system(self, coefs, knots, deg, bc, matching):
        """
        System of non-linear equations for finte-element method.
        
        Arguments:
            
            coefs: (array) (2 * N + 2,) array of B-spline coefficients. The 
                   first N + 1 coefficients are used to constuct the B-spline 
                   approximation of the equilibrium firm size function theta(x).
                   The second N + 1 coefficients are used to construct the 
                   B-spline approximation of the equilibrium matching function 
                   mu(x). 
                   
            knots: (array) (2, N) array of B-spline knots. The first row of 
                   knots is used to constuct the B-spline approximation of the 
                   equilibrium firm size function theta(x). The second row of 
                   knots is used to construct the B-spline approximation of the 
                   equilibrium matching function mu(x). 
                   
            deg:   (int) Degree of desired B-spline. Must satisfy 1 <= deg <= 5.
                
            bc:    (list) List of tuples defining the boundary conditions. 
                   Tuples should be (x, mu(x)) pairs which pin down the value of
                   equilibrium matching function at the end-points.                    
        Returns:
            
            out: (array) Array of residuals.
            
        """
        N = knots.shape[1]
        coefs = coefs.reshape((2, N + 1))
        
        # lower boundary condition
        x_lower = bc[0][0]
        y_lower = bc[0][1]

        # upper boundary condition
        x_upper = bc[1][0]
        y_upper = bc[1][1]
        
        # compute the equilibrium matching function at the boundary-points
        mu = lambda x: self.bspline(x, knots[1], coefs[1], deg)
        boundary_conditions = np.array([mu(x_lower) - y_lower, 
                                        mu(x_upper) - y_upper])
        
        # compute the value of the residual for given set of coefs
        out = np.hstack((boundary_conditions, 
                         self.residual(knots, knots, coefs, deg, matching)))
                         
        return out

    def solve(self, knots, init_coefs, deg, ext, bc, matching='pam', 
              method='hybr'):
        """
        Solves for the equilibrium functions theta(x) and mu(x) using a finite- 
        elements approach.
        
        Arguments:
                   
            knots:      (array) (2, N) array of knots.
            
            init_coefs: (array) (2, N + 1) array of coefficients representing a 
                        guess of the optimal set of coefficients.
                        
            bc:         (list) List of tuples defining the boundary conditions. 
                        Tuples should be (x, mu(x)) pairs which pin down the 
                        value of the equilibrium matching function at its 
                        end-points.
              
        If the boundary value problem was successfully solved, then the 
        equilibrium functions theta(x), mu(x), w(x) and pi(x) are stored in the
        model's equilibrium dictionary. If a solution was not found, then the 
        result object is returned for inspection.
                                  
        """        
        # solve the system of non-linear equations 
        res = optimize.root(self.system, init_coefs.flatten(), 
                            args=(knots, deg, bc, matching), method=method)
        
        # if solution has been found, modify the equilibrium attribute 
        if res.success == True:
            
            # extract and store the solution
            self.coefs = res.x.reshape(init_coefs.shape)
            self.knots = knots
            self.deg   = deg
            
            # approximate equilibrium firm size and matching functions
            theta   = lambda x: self.bspline(x, self.knots[0], self.coefs[0], 
                                             self.deg, ext)
            mu      = lambda x: self.bspline(x, self.knots[1], self.coefs[1],  
                                             self.deg, ext)  
            
            # approximate equilibrium wages and profits
            w  = lambda x: self.model.get_wages(x, [theta(x), mu(x)])
            pi = lambda x: self.model.get_profits(x, [theta(x), mu(x)])
            
            # modify the equilibrium attribute
            self.model.equilibrium['theta']   = theta
            self.model.equilibrium['mu']      = mu
            self.model.equilibrium['wages']   = w
            self.model.equilibrium['profits'] = pi 
            
        else:
            return res
    
    def plot_residual(self, N=1000, matching='pam'):
        """
        Generates a plot of the finite-element residuals for theta(x) and mu(x).
        
        Arguments:
            
            N: (int) Number of points to plot.
            
        """    
        ax0 = plt.subplot(211) 
        ax1 = plt.subplot(212) 

        x_lower = self.model.workers.lower_bound
        x_upper = self.model.workers.upper_bound
        x_grid = np.linspace(x_lower, x_upper, 2 * N).reshape((2, N)) 
        
        # define the residual function
        resid = self.residual(x_grid, self.knots, self.coefs, self.deg, matching)        
        
        # plot the theta residual
        ax0.plot(x_grid[0], resid[:N])
        ax0.set_ylabel(r'$R_{\theta}$', rotation='horizontal', fontsize=15)
        ax0.set_title(r'Finite-element residuals for $\theta(x)$ and $\mu(x)$.',
                      fontsize=15, family='serif')
                      
        # plot the mu residual
        ax1.plot(x_grid[1], resid[N:])
        ax1.set_xlabel('$x$', fontsize=15)
        ax1.set_ylabel(r'$R_{\mu}$', rotation='horizontal', fontsize=15)

###################### Functions for analyzing equilibria ######################

def inverse_matching_function(y):
     pass
            
def get_realized_productivity():
    pass
    
def get_gini_coefficient(arr):
    """Computes the Gini coefficient of the array x.

    Arguments:

        arr: (array-like) Will be converted to a Pandas Series object if 
             necessary.

    Returns:

        G: (scalar) the Gini coefficient 
        
    """
    # convert to Pandas series if necessary
    if not isinstance(arr, pd.Series):
        arr = pd.Series(arr)
    else:
        pass
        
    # make sure that x is sorted in ascending order
    sorted_arr = arr.order()

    # compute the Gini coefficient
    cumsum_arr = sorted_arr.cumsum()
    B = np.sum(cumsum_arr) / (cumsum_arr.max() * arr.size)
    gini = 1 + (1.0 / arr.size) - 2 * B
    
    return gini
