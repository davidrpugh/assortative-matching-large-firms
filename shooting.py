"""
Solves a model using forward shooting.

@author : David R. Pugh

"""
from __future__ import division
import numpy as np
import pandas as pd
from quantecon import ivp
import sympy as sym


import solvers

# represent endogenous variables mu and theta as a deferred vector
V = sym.DeferredVector('V')


class ShootingSolver(solvers.Solver):
    """Represent a Solver that implements a forward-shooting algorithm."""

    __numeric_jacobian = None

    __numeric_system = None

    __ivp = None

    @property
    def _numeric_jacobian(self):
        """Vectorized function for numerical evaluation of model Jacobian."""
        if self.__numeric_jacobian is None:
            self.__numeric_jacobian = self._lambdify(self._symbolic_jacobian)
        return self.__numeric_jacobian

    @property
    def _raw_residuals(self):
        """Raw model residuals stored in a `pandas.DataFrame`."""
        if self.model.assortativity == 'positive':
            traj = self._solution[::-1]
        else:
            traj = self._solution

        # compute the residuals
        xi = np.linspace(traj[0, 0], traj[-1, 0], 10 * traj.shape[0])
        resids_arr = self.ivp.compute_residual(traj[:, :3], xi, k=5, ext=2)

        # convert to a data frame
        resids_arr[:, 0] = xi
        col_names = ['x', r'$\hat{\mu}(x)$', r'$\hat{\theta}(x)$']
        df = pd.DataFrame(resids_arr, columns=col_names)

        return df.set_index('x')

    @property
    def _numeric_system(self):
        """Vectorized function for numerical evaluation of model system."""
        if self.__numeric_system is None:
            self.__numeric_system = self._lambdify(self._symbolic_system)
        return self.__numeric_system

    @property
    def _symbolic_equations(self):
        """List of symbolic expressions for right-hand side of ODE system."""
        return [self.model.matching.mu_prime, self.model.matching.theta_prime]

    @property
    def _symbolic_jacobian(self):
        """Symbolic Jacobian matrix for the ODE system."""
        return self._symbolic_system.jacobian([V[0], V[1]])

    @property
    def _symbolic_system(self):
        """Symbolic matrix defining the right-hand side of ODE system."""
        system = sym.Matrix(self._symbolic_equations)
        return system.subs(self._symbolic_change_of_vars)

    @property
    def interpolated_solution(self):
        """
        Interpolated solution to the model

        Notes
        -----
        The solution array is interpolated using 5-order parameteric, B-spline
        interpolation (with no extrapolation).

        """
        # need to reverse the direction of the _solution array
        if self.model.assortativity == 'positive':
            traj = self._solution[::-1]
        else:
            traj = self._solution

        xi = np.linspace(traj[0, 0], traj[-1, 0], 10 * traj.shape[0])
        interp_soln = self.ivp.interpolate(traj, xi, k=5, ext=2)

        # convert to a data frame
        col_names = ['x', r'$\hat{\mu}(x)$', r'$\hat{\theta}(x)$', '$\hat{w}(x)$', r'$\hat{\pi}(x)$']
        df = pd.DataFrame(interp_soln, columns=col_names)

        return df.set_index('x')

    @property
    def ivp(self):
        r"""
        An instance of the `quantecon.ivp.IVP` class representing the model as
        an initial value problem (IVP).

        :getter: Return an instance of the `ivp.IVP` class.
        :type: ivp.IVP

        """
        if self.__ivp is None:
            self.__ivp = ivp.IVP(self.evaluate_rhs, self.evaluate_jacobian)
        return self.__ivp

    @property
    def residuals(self):
        r"""
        Model residuals for :math:`\hat{\mu}(x)` and :math:`\hat{\theta}(x)`.

        :getter: Return the current model residuals.
        :type: pandas.DataFrame

        """
        normed_resids = (np.abs(self._raw_residuals[[0, 1]]) /
                         self.interpolated_solution[[0, 1]])
        return normed_resids

    def _clear_cache(self):
        """Clear cached functions used for numerical evaluation."""
        super(ShootingSolver, self)._clear_cache()
        self.__numeric_jacobian = None
        self.__numeric_system = None
        self.__ivp = None

    def _converged_firms(self, tol):
        """Check whether solution component for firms has converged."""
        mu = self.ivp.y[0]
        y_lower = self.model.firms.lower

        if abs(mu - y_lower) / mu <= tol:  # use relative values!
            converged = True
        else:
            converged = False

        return converged

    def _converged_workers(self, tol):
        """Check whether solution component for workers has converged."""
        x = self.ivp.t
        x_lower = self.model.workers.lower
        x_upper = self.model.workers.upper

        if self.model.assortativity == 'positive':
            if abs(x - x_lower) / x <= tol:  # use relative values!
                converged = True
            else:
                converged = False
        else:
            if abs(x - x_upper) / x <= tol:  # use relative values!
                converged = True
            else:
                converged = False

        return converged

    def _exhausted_firms(self, tol):
        """Check whether firms have been exhausted."""
        mu = self.ivp.y[0]
        y_lower = self.model.firms.lower

        if (mu - y_lower) / mu < -tol:  # use relative values!
            exhausted = True
        else:
            exhausted = False

        return exhausted

    def _guess_firm_size_upper_too_low(self, bound, tol):
        """Check whether guess for upper bound for firm size is too low."""
        theta = self.ivp.y[1]
        return abs(theta - bound) / theta <= tol  # use relative values!

    def _reset_solution(self, firm_size):
        """Reset initial condition and re-initialze solution."""
        x_lower, x_upper = self.model.workers.lower, self.model.workers.upper
        y_upper = self.model.firms.upper
        initial_V = np.array([y_upper, firm_size])

        if self.model.assortativity == 'positive':
            self.ivp.set_initial_value(initial_V, x_upper)
            wage = self.evaluate_wage(x_upper, initial_V)
            profit = self.evaluate_profit(x_upper, initial_V)
            self._solution = np.hstack((x_upper, initial_V, wage, profit))
        else:
            self.ivp.set_initial_value(initial_V, x_lower)
            wage = self.evaluate_wage(x_lower, initial_V)
            profit = self.evaluate_profit(x_lower, initial_V)
            self._solution = np.hstack((x_lower, initial_V, wage, profit))

    def _update_initial_guess(self, lower, upper):
        """Use bisection method to arrive at new initial guess."""
        err_mesg = 'Upper and lower bounds are identical: check solver tols!'
        assert (upper - lower) / upper > np.finfo('float').eps, err_mesg
        guess = 0.5 * (lower + upper)
        return guess

    def _update_solution(self, step_size):
        """Update the solution array after an integration step."""
        if self.model.assortativity == 'positive':
            self.ivp.integrate(self.ivp.t - step_size)
            x, V = self.ivp.t, self.ivp.y
        else:
            self.ivp.integrate(self.ivp.t + step_size)
            x, V = self.ivp.t, self.ivp.y

        assert V[1] > 0.0, "Firm size should be non-negative!"

        # update the putative equilibrium solution
        wage = self.evaluate_wage(x, V)
        profit = self.evaluate_profit(x, V)
        step = np.hstack((x, V, wage, profit))
        self._solution = np.vstack((self._solution, step))

    def evaluate_jacobian(self, x, V):
        r"""
        Numerically evaluate model Jacobian.

        Parameters
        ----------
        x : float
            Value for worker skill (i.e., the independent variable).
        V : numpy.array (shape=(2,))
            Array of values for the dependent variables with ordering:
            :math:`[\mu, \theta]`.

        Returns
        -------
        jac : numpy.array (shape=(2,2))
            Jacobian matrix of partial derivatives.

        """
        jac = self._numeric_jacobian(x, V, *self.model.params.values())
        return jac

    def evaluate_rhs(self, x, V):
        r"""
        Numerically evaluate right-hand side of the ODE system.

        Parameters
        ----------
        x : float
            Value for worker skill (i.e., the independent variable).
        V : numpy.array (shape=(2,))
            Array of values for the dependent variables with ordering:
            :math:`[\mu, \theta]`.

        Returns
        -------
        rhs : numpy.array (shape=(2,))
            Right hand side of the system of ODEs.

        """
        rhs = self._numeric_system(x, V, *self.model.params.values()).ravel()
        return rhs

    def _compute_step_sizes(self, number_knots, knots):
        """Computes an array of implied step sizes given some knot sequence."""
        # expected bounds on the knots sequence
        x_lower = self.model.workers.lower
        x_upper = self.model.workers.upper

        if (number_knots is not None) and (knots is None):
            step_size = (x_upper - x_lower) / (number_knots - 1)
            step_sizes = np.repeat(step_size, number_knots - 1)
        elif (number_knots is None) and (knots is not None):
            #assert knots[0] == x_lower
            #assert knots[-1] == x_upper
            step_sizes = np.diff(knots, 1)
        else:
            raise ValueError("Either 'number_knots' or 'knots' must be specified!")
        
        if self.model.assortativity == 'positive':
            step_sizes = step_sizes[::-1]

        return step_sizes

    def solve(self, guess_firm_size_upper, tol=1e-6, number_knots=100, knots=None,
              integrator='dopri5', message=False, **kwargs):
        """
        Solve for assortative matching equilibrium.

        Parameters
        ----------
        guess_firm_size_upper : float
            Upper bound on the range of possible values for the initial
            condition for firm size.
        tol : float (default=1e-6)
            Convergence tolerance.
        number_knots : int (default=100)
            Number of knots to use in approximating the solution. The number of
            knots determines the step size used by the ODE solver.
        integrator: string (default='dopri5')
            Integrator to use in appoximating the solution. Valid options are:
            'dopri5', 'lsoda', 'vode', 'dop853'. See `scipy.optimize.ode` for
            complete description of each solver.
        message : boolean (default=False)
            Flag indicating whether or not to print progress messages.
        **kwargs : dict
            Dictionary of optional, solver specific, keyword arguments. setter
            `scipy.optimize.ode` for details.

        Notes
        -----
        Rather than returning a result, this method modifies the `_solution`
        attribute of the `Solver` class. To final solution is stored in the
        `solution`attribute as a `pandas.DataFrame`.

        """

        # relevant bounds
        x_lower = self.model.workers.lower
        x_upper = self.model.workers.upper

        # initialize integrator
        self.ivp.set_integrator(integrator, **kwargs)

        # initialize the solution
        firm_size_lower = 0.0
        firm_size_upper = guess_firm_size_upper
        guess_firm_size = 0.5 * (firm_size_upper + firm_size_lower)
        self._reset_solution(guess_firm_size)

        # get the array of step sizes
        step_sizes = self._compute_step_sizes(number_knots, knots)
        idx = 0
        #assert step_sizes > 0

        while self.ivp.successful():

            self._update_solution(step_sizes[idx])

            if self._converged_workers(tol) and self._converged_firms(tol):
                self._validate_solution(self._solution, tol)
                mesg = "Success! All workers and firms are matched"
                print(mesg)
                break

            elif (not self._converged_workers(tol)) and self._exhausted_firms(tol):
                if message:
                    mesg = ("Exhausted firms: initial guess of {} for firm " +
                            "size is too low.")
                    print(mesg.format(guess_firm_size))
                idx = 0
                firm_size_lower = guess_firm_size

            elif self._converged_workers(tol) and self._exhausted_firms(tol):
                if message:
                    mesg = ("Exhausted firms: initial guess of {} for firm " +
                            "size was too low!")
                    print(mesg.format(guess_firm_size))
                idx = 0
                firm_size_lower = guess_firm_size

            elif self._converged_workers(tol) and (not self._exhausted_firms(tol)):
                if message:
                    mesg = ("Exhausted workers: initial guess of {} for " +
                            "firm size is too high!")
                    print(mesg.format(guess_firm_size))
                idx = 0 # reset step size
                firm_size_upper = guess_firm_size

            else:
            	idx += 1
                continue

            guess_firm_size = self._update_initial_guess(firm_size_lower,
                                                         firm_size_upper)
            self._reset_solution(guess_firm_size)

            # reset solution should not be too close to guess_firm_size_upper!
            err_mesg = ("Failure! Need to increase initial guess for upper " +
                        "bound on firm size!")
            assert not self._guess_firm_size_upper_too_low(guess_firm_size_upper, tol), err_mesg
