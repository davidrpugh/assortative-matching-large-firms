"""
Module implementing an orthogonal collocation solver.

@author : David R. Pugh
@date : 2014-12-05

"""
from __future__ import division
import numpy as np
from scipy import optimize

import solvers


class OrthogonalCollocation(solvers.Solver):
    """Class representing an orthogonal collocation solver."""

    __coefs = None

    __nodes = None

    def __init__(self, model):
        """Create an instance of the OrthogonalCollocation class."""
        super(OrthogonalCollocation, self).__init__()
        self.kind = "Chebyshev"

    @property
    def _coefs_mu(self):
        return self.__coefs_mu

    @_coefs_mu.setter
    def _coefs_mu(self, value):
        self.__coefs_mu = value

    @property
    def _coefs_theta(self):
        return self.__coefs_theta

    @_coefs_theta.setter
    def _coefs_theta(self, value):
        self.__coefs_theta = value

    @property
    def _domain(self):
        return [self.model.workers.lower, self.model.workers.upper]

    @property
    def kind(self):
        """
        Kind of orthogonal polynomials to use in approximating the solution.

        :getter: Return the current kind of orthogonal polynomials.
        :setter: Set a new kind of orthogonal polynomials.
        :type: string

        """
        return self._kind

    @kind.setter
    def kind(self, kind):
        """Set a new kind of orthogonal polynomials."""
        self._kind = self._validate_kind(kind)

    @property
    def orthogonal_poly_mu(self):
        r"""
        Orthogonal polynomial approximation of the assignment function,
        :math:`\mu(x)`.

        :getter: Return the orthogonal polynomial approximation.
        :type: numpy.polynomial.Polynomial

        """
        if self.kind == 'Chebyshev':
            poly = np.polynomial.Chebyshev(self._coefs_mu, self._domain)
        else:
            assert False, "Invalid value specified for the 'kind' attribute."
        return poly

    @property
    def orthogonal_poly_theta(self):
        r"""
        Orthogonal polynomial approximation of the firm size function,
        :math:`\theta(x)`.

        :getter: Return the orthogonal polynomial approximation.
        :type: numpy.polynomial.Polynomial

        """
        if self.kind == 'Chebyshev':
            poly = np.polynomial.Chebyshev(self._coefs_theta, self._domain)
        else:
            assert False, "Invalid value specified for the 'kind' attribute."
        return poly

    @staticmethod
    def _validate_kind(kind):
        """Validate the model attribute."""
        valid_kinds = ['Chebyshev']
        if not isinstance(kind, str):
            mesg = ("Attribute 'kind' must have type str, not {}.")
            raise AttributeError(mesg.format(kind.__class__))
        elif kind not in valid_kinds:
            mesg = "Attribute 'kind' must be one of {}."
            raise AttributeError(mesg.format(valid_kinds))
        else:
            return kind

    def evaluate_residual_mu(self, x):
        V = np.hstack((self.orthogonal_poly_mu(x),
                       self.orthogonal_poly_theta(x)))
        mu_residual = (self.orthogonal_poly_mu.deriv()(x) -
                       self.evaluate_rhs_mu(x, V))
        return mu_residual

    def evaluate_residual_theta(self, x):
        V = np.hstack((self.orthogonal_poly_mu(x),
                       self.orthogonal_poly_theta(x)))
        theta_residual = (self.orthogonal_poly_theta.deriv()(x) -
                          self.evaluate_rhs_theta(x, V))
        return theta_residual

    def evaluate_residuals(self, x):
        residuals = np.hstack((self.evaluate_rhs_mu(x),
                               self.evaluate_residual_theta(x)))
        return residuals

    def evaluate_rhs_mu(self, x, V):
        raise NotImplementedError

    def evaluate_rhs_theta(self, x, V):
        raise NotImplementedError

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
