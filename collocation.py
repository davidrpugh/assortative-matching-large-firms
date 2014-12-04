
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
