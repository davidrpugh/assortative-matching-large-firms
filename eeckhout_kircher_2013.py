from __future__ import division
import numpy as np

from pyeconomics.ode.solvers import *
from pyeconomics.ode.integrators import *

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

class Model(BVP):
    """
    Class for solving and simulating the Eeckhout and Kircher (2013) model.

    ABSTRACT: In large firms, management resolves a trade-off between hiring
    more versus better workers.  The span of control or size if therefore 
    intimately  intertwined with the sorting pattern. Span of control is at the 
    center of  many studies in macroeconomics, comparisons of factor  
    productivity, trade, and labor.  With hetergeneous workers, we analyze the 
    worker assignment, firm size, and wages.  The pattern of sorting between  
    workers and firms is governed by an intuitive cross-margin-complementarity 
    condition that captures the complementarities between qualities (of workers 
    quantities (of the work force and firm resources). A simple system of two 
    and firms), and differential equations determines the equilibrium 
    allocation, firm size, and wages.  We can analyze the impact of 
    technological change: skill-biased  change affects wages and increases the 
    skill premium; quantity-biased  change affects firm size, especially of 
    productive firms.  We also introduce search frictions and investigate how  
    unemployment varies across skills and vacancies vary across firm size.

    """
    def __init__(self, params, workers, firms, F, derivatives, PAM=True):
        """
        Initializes an object representing the Eeckhout and Kircher (2013) 
        model of assortative matching with large firms.

        Attributes:

            params:      (dict) Dictionary of model parameters.
            
            workers:     (object) An object of the Workers class describing the 
                         distribution of workers.
            
            firms:       (object) An object of the Firms class describing the 
                         distribution of firms.
            
            F:           (function) Output, F, is a function of the form
            
                             f(x, vec, params) = F(x, y, (l / r), 1, params)
                     
                         where x is worker type, y is firm productivity, y, l is 
                         the quantity of labor of type x, and r is the fraction 
                         of proprietary resources devoted to a worker of type x. 
                         Final argument of f should be a dictionary of model 
                         parameters. 
            
            derivatives: (dict) Dictionary of callable functions defining the 
                         partial and cross partial derivatives necessary to 
                         construct the system of ODEs describing the assortative
                         matching equilibrium and its Jacobian (if necessary). 
                         
            PAM:         (boolean) If True, then you are interested in a 
                         positive assortative matching (PAM) equilibrium; if 
                         False, then you are interested in a negative 
                         assortative matching (NAM) equilibrium.

        """
        # initialize attributes
        self.workers     = workers
        self.firms       = firms 
        self.F           = F       
        self.derivatives = derivatives
        self.PAM         = PAM 
        
        if self.PAM == True:
            f = self.get_pam_system
            jac = None
        elif self.PAM == False:
            f = self.get_nam_system
            jac = None
        else:
            raise ValueError
            
        super(Model, self).__init__(f, jac, params)
        
    def get_pam_system(self, xn, vec, params):
        """
        2D system of differential equations describing the behavior of an 
        equilibrium with positive assortative matching between workers and 
        firms.

        Arguments:

            xn:     (float) Worker type/skill.
            vec:    (array) Vector of endogenous variables, [mu, theta].
            params: (dict) Dictionary of model parameters.
            
        Returns:
            
            out: (array) Vector of values [mu', theta'].

        """
        out = np.array([self.__mu_prime(xn, vec, params),
                        self.__theta_prime(xn, vec, params)], dtype='float')
        return out

    def get_nam_system(self, xn, vec, params):
        """
        2D system of differential equations describing the behavior of an 
        equilibrium with negative assortative matching between workers and 
        firms.

        Arguments:

            xn:     (float) Worker type/skill.
            vec:    (array) Vector of endogenous variables, [mu, theta].
            params: (dict) Dictionary of model parameters.
            
        Returns:
            
            out: (array) Vector of values [mu', theta'].

        """
        out = np.array([-self.__mu_prime(xn, vec, params),
                        -self.__theta_prime(xn, vec, params)], dtype='float')
        return out
        
    def __theta_prime(self, xn, vec, params):
        """ODE governing the evolution of factor intensity."""
        # span of control complementarity
        Fyl = self.derivatives['Fyl']

        # managerial resource complementarity 1
        Fxr = self.derivatives['Fxr']

        # workers resource complementarity
        Flr = self.derivatives['Flr']

        out = (self.__H(xn, vec, params) * Fyl(xn, vec, params) -  
               Fxr(xn, vec, params)) / Flr(xn, vec, params)
               
        return out

    def __mu_prime(self, xn, vec, params):
        """ODE governing the evolution of matching function."""
        return self.__H(xn, vec, params) / vec[1]
        
    def __H(self, xn, vec, params):
        """
        Ratio of density functions of worker skill and firm productivity 
        (evaluated along the equilibrium path!).

        Arguments:

            xn:     (float) Worker type/skill.
            vec:    (array) Vector of endogenous variables, [mu, theta].
            params: (dict) Dictionary of model parameters.
            
        Returns:
            
            out: (array) Ratio of worker-firm densities.

        """
        out = self.workers.scaled_pdf(xn) / self.firms.scaled_pdf(vec[0])
        return out
    
    def get_wages(self, xn, vec, params):
        """
        Equilibrium wage paid to a worker of type x.

        Arguments:

            xn:     (float) Worker type/skill.
            vec:    (array) Vector of endogenous variables, [mu, theta].
            params: (dict) Dictionary of model parameters.
            
        Returns:
            
            ftheta: (float) Real wage paid to a worker of type x. 

        """
        # labor is paid its marginal product
        ftheta = self.derivatives['ftheta'](xn, vec, params)
        
        return ftheta
        
    def get_profits(self, xn, vec, params):
        """
        Equilibrium profits paid to a firm with productivity mu(x).

        Arguments:

            xn:     (float) Worker type/skill.
            vec:    (array) Vector of endogenous variables, [mu, theta].
            params: (dict) Dictionary of model parameters.
            
        Returns:
            
            profits: (float) Profits paid to firm with productivity mu(x).

        """
        # profits are revenue less labor costs
        revenue = self.F(xn, vec, params) 
        costs   = vec[1] * self.get_equilibrium_wages(xn, vec, params) 
        profits = revenue - costs
        
        return profits
        
    def set_firm_size_distribution(self, traj, bbox=[None, None], k=3):
        """Use B-spline interpolation to construct a callable function 
        returning a firm's size as a function of its productivity, y.
        
        Arguments:
            
            traj: (array) Simulated solution trajectory approximating the 
                  assortative matching equilibrium functions.
            
            bbox: (array-like) Length 2-sequence specifying the boundary of the
                  approximation interval. Default is [None, None] which implies
                  bounds will be set to [theta[0], theta[-1]].
                  
            k:    (int) Degree of B-spline approximation to use. Must be <= 5.
        
        """
        y     = traj[:, 1]
        theta = traj[:, 2]
        
        # interpolate to approximate y_theta
        y_theta = interpolate.UnivariateSpline(theta, y, bbox=bbox, k=k, s=0)

        # modify the firms' size distribution attribute
        firm_size_pdf = lambda theta: self.firms.dist.pdf(y_theta(theta))
        self.firms.size_distribution = firm_size_pdf
        
    def set_firm_profits_distribution(self, traj, bbox=[None, None], k=3):
        """Use B-spline interpolation to construct a callable function 
        returning a firm's profits as a function of its productivity, y.
        
        Arguments:
            
            traj: (array) Simulated solution trajectory approximating the 
                  assortative matching equilibrium functions.
            
            bbox: (array-like) Length 2-sequence specifying the boundary of the
                  approximation interval. Default is [None, None] which implies
                  bounds will be set to [theta[0], theta[-1]].
                  
            k:    (int) Degree of B-spline approximation to use. Must be <= 5.
        
        """
        # compute firm profits 
        x       = traj[:, 0]
        vec     = traj[:, 1:]
        profits = [self.get_profits(x, vec, params) for i in range(len(traj))]
        
        # extract y = mu(x)
        y       = traj[:, 1]
        
        # interpolate to approximate y(pi)
        y_pi = interpolate.UnivariateSpline(profits, y, bbox=bbox, k=k, s=0)

        # modify the firms' size distribution attribute
        firm_pi_pdf = lambda pi: self.firms.dist.pdf(y_pi(pi))
        self.firms.size_distribution = firm_pi_pdf