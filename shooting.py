from scipy import integrate

from traits.api import Float, HasPrivateTraits, Instance, Property, Str, Tuple

import sandbox

class ShootingSolver(HasPrivateTraits):
    """Class representing a simple shooting solver."""

    _initial_condition = Property(Tuple)

    _integrator = Property(Instance(integrate.ode), 
                           depends_on=['_initial_condition', 'model'])

    # need to validate this trait!
    integrator = Str('dopri5')
    
    model = Instance(sandbox.Model)

    theta0 = Float(1.0)

    def _get__initial_condition(self):
        """Initial condition for the solver."""
        
        x_lower = self.model.workers.lower_bound
        x_upper = self.model.workers.upper_bound
        y_upper = self.model.firms.upper_bound

        if self.model.matching =='pam':
            init = (np.array([self.theta0, y_upper]), x_upper)
        else:
            init = (np.array([self.theta0, y_upper]), x_lower)

        return init

    def _get__integrate(self):
        pass

    def _solve_pam(self, theta0, tol=1e-2, N=1e3, deg=1, mesg=False, 
                   max_iter=1e6, integrator='dopri5', **kwargs):
        """Uses a forward shooting algorithm to solve for a PAM euilibrium."""

        # range of worker types 
        xl = self.model.workers.lower_bound
        xu = self.model.workers.upper_bound
            
        # range of worker types 
        yl = self.model.firms.lower_bound 
        yu = self.model.firms.upper_bound
    
        # initial condition for the solver 
        init = np.array([theta0, yu])

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
        solver.set_integrator(integrator, **kwargs)
        solver.set_initial_value(init, xu)
                
        ##### Forward shooting algorithm #####
    
        # initialize putative equilibrium path
        init_wages   = self.model.get_wages(xu, init)
        init_profits = self.model.get_profits(xu, init)
        path         = np.hstack((xu, init, init_wages, init_profits)) 

        # initialize a counter
        num_iter = 0

        # initial feasible range for theta0 
        theta_h = 2.0 * theta0
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

                # theta0 too low  
                theta_l = init[0]
                init[0] = (theta_h + theta_l) / 2
                if mesg == True:
                    print 'Negative firm size! Guess for theta0 was too low!'
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

                # ...check if exhausted all firms (i.e., theta0 too low!)
                elif (solver.y[1] - yl) < -tol:
                    theta_l = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print ('You have run out of firms! Guess for theta0' + 
                               ' was too low!')
                        print 'Most recent step:', path[-1,:]
                        print 'New initial condition is', init[0]
                        print ''
                    solver.set_initial_value(init, xu)
                
                    # reset the putative equilibrium path
                    init_wages = self.model.get_wages(xu, init)
                    init_profits = self.model.get_profits(xu, init)
                    path = np.hstack((xu, init, init_wages, init_profits))
                
                # ...else, theta0 too high!
                else:
                    theta_h = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print('You have run out of workers, but there are ' + 
                              'still firms around and profits are non-zero.' +
                              ' Guess for theta0 was too high!') 
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

                # ...check if exhausted all workers (i.e., theta0 too high!)
                elif (solver.t - xl) < -tol:
                    theta_h = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print ('You have run out of workers! Guess for' +
                               ' theta0 was too high!') 
                        print 'Most recent step:', path[-1,:]
                        print 'New initial condition is', init[0]
                        print ''
                    solver.set_initial_value(init, xu)
                
                    # reset the putative equilibrium path
                    init_wages = self.model.get_wages(xu, init)
                    init_profits = self.model.get_profits(xu, init)
                    path = np.hstack((xu, init, init_wages, init_profits))
                
                # ...else, theta0 too low!
                else:
                    theta_l = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print('You have run out of firms, but there are ' + 
                              'still workers around and wages are non-zero.' +
                              ' Guess for theta0 was too low!')
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

                # theta0 too high!  
                else:
                    theta_h = init[0]
                    init[0] = (theta_h + theta_l) / 2
                    if mesg == True:
                        print('There are still workers and firms around, ' + 
                              'but wages are zero! Guess for theta0 was too' +
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

                # theta0 too low!
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


    def _solve_nam(self, theta0, tol=1e-2, N=1e3, deg=1, mesg=False, 
                   max_iter=1e6, integrator='dopri5', **kwargs):
        raise NotImplementedError

    def solve(self, theta0, tol=1e-2, N=1e3, deg=1, mesg=False, 
              max_iter=1e6, integrator='dopri5', **kwargs):
        """
        Solve for the equilibrium of the Eeckhout and Kircher (2013) model 
        using a Forward Shooting algorithm. For details on the forward shooting 
        algorithm see Judd (1992) p. 357.

        Arguments:

        theta0:    Initial guess for the size of the most productive firm.
        
        tol:        How close to the shooting target is "close enough."
        
        N:          Number of points of the solution path to compute.
        
        deg:        Degree of B-spline approximation to use. Must satisfy
                    1 <= deg <= 5.
                    
        matching:   One of either 'pam' or 'nam' depending on whether or not you
                    want an equilibrium with positive or negative assortative 
                    matching.
        
        mesg:       Do you want to print messages indicating progress
                    towards convergence? Default is False.
        
        integrator: Solution method for ODE solver.  Must be one of 'lsoda', 
                    'vode', zvode', 'dopri5', 'dop853'.  See scipy.integrate.ode 
                    for details (including references for algorithms).
        
        **kwargs:   Additional arguments to pass to the ODE solver.  
                    See scipy.integrate.ode for details.
        
        """
        # which or either positive or negative assortative matching?
        if self.model.matching == 'pam':
            self._shoot_pam(theta0, tol, N, deg, mesg, max_iter, integrator, 
                            **kwargs)

        else:
            self._shoot_nam(theta0, tol, N, deg, mesg, max_iter, integrator, 
                            **kwargs)
        
    


if __name__ == '__main__':
    import numpy as np
    import sympy as sp

    from inputs import Input
    from sandbox import Model

    params = {'nu':0.89, 'kappa':1.0, 'gamma':0.54, 'rho':0.24, 'A':1.0}

    sp.var('A, kappa, nu, x, rho, y, l, gamma, r')
    F = r * A * kappa * (nu * x**rho + (1 - nu) * (y * (l / r))**rho)**(gamma / rho)

    # suppose worker skill is log normal
    sp.var('x, mu, sigma')
    skill_cdf = 0.5 + 0.5 * sp.erf((sp.log(x) - mu) / sp.sqrt(2 * sigma**2))
    worker_params = {'mu':0.0, 'sigma':1}
    x_lower, x_upper, x_measure = 1e-3, 5e1, 0.025
    
    workers = Input(distribution=skill_cdf,
                    lower_bound=x_lower,
                    params=worker_params,
                    upper_bound=x_upper,
                   )

    # suppose firm productivity is log normal
    sp.var('x, mu, sigma')
    productivity_cdf = 0.5 + 0.5 * sp.erf((sp.log(x) - mu) / sp.sqrt(2 * sigma**2))
    firm_params = {'mu':0.0, 'sigma':1}
    y_lower, y_upper, y_measure = 1e-3, 5e1, 0.025
    
    firms = Input(distribution=productivity_cdf,
                  lower_bound=x_lower,
                  params=firm_params,
                  upper_bound=x_upper,
                  )

    # create an instance of the Model class
    model = Model(firms=firms,
                  matching='PAM',
                  output=F,
                  params=params,
                  workers=workers
                  )

    shooting = ShootingSolver(integrator='dopri5',
                              model=model)
