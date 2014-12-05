import numpy as np
from scipy import integrate
import sympy as sym

import solvers

# represent endogenous variables mu and theta as a deferred vector
V = sym.DeferredVector('V')


class ShootingSolver(solvers.Solver):
    """Solves a model using forward shooting."""

    __numeric_jacobian = None

    __numeric_system = None

    __integrator = None

    @property
    def _numeric_jacobian(self):
        """
        Vectorized function for numerical evaluation of model Jacobian.

        :getter: Return current function for evaluating the Jacobian.
        :type: function

        """
        if self.__numeric_jacobian is None:
            self.__numeric_jacobian = sym.lambdify(self._symbolic_args,
                                                   self._symbolic_jacobian,
                                                   self._modules)
        return self.__numeric_jacobian

    @property
    def _numeric_system(self):
        """
        Vectorized function for numerical evaluation of model system.

        :getter: Return current function for evaluating the system.
        :type: function

        """
        if self.__numeric_system is None:
            self.__numeric_system = sym.lambdify(self._symbolic_args,
                                                 self._symbolic_system,
                                                 self._modules)
        return self.__numeric_system

    @property
    def _symbolic_args(self):
        """
        Symbolic arguments used when lambdifying symbolic Jacobian and system.

        :getter: Return the list of symbolic arguments.
        :type: list

        """
        return self._symbolic_variables + self._symbolic_params

    @property
    def _symbolic_equations(self):
        """
        Symbolic expressions defining the right-hand side of a system of ODEs.

        :getter: Return the list of symbolic expressions.
        :type: list

        """
        return [self.model.matching.mu_prime, self.model.matching.theta_prime]

    @property
    def _symbolic_jacobian(self):
        """
        Symbolic expressions defining the Jacobian of a system of ODEs.

        :getter: Return the symbolic Jacobian.
        :type: sympy.Basic

        """
        return self._symbolic_system.jacobian([V[0], V[1]])

    @property
    def _symbolic_params(self):
        """
        Symbolic parameters passed as arguments when lambdifying symbolic
        Jacobian and system.

        :getter: Return the list of symbolic parameter arguments.
        :type: list

        """
        return sym.var(list(self.model.params.keys()))

    @property
    def _symbolic_system(self):
        """
        Symbolic matrix defining the right-hand side of a system of ODEs.

        :getter: Return the symbolic matrix.
        :type: sympy.Matrix

        """
        system = sym.Matrix(self._symbolic_equations)
        return system.subs({'mu': V[0], 'theta': V[1]})

    @property
    def _symbolic_variables(self):
        """
        Symbolic variables passed as arguments when lambdifying symbolic
        Jacobian and system.

        :getter: Return the list of symbolic variable arguments.
        :type: list

        """
        return [self.model.workers.var, V]

    @property
    def integrator(self):
        """
        Integrator for solving a system of ordinary differential equations.

        :getter: Return the current integrator.
        :type: scipy.integrate.ode

        """
        if self.__integrator is None:
            self.__integrator = integrate.ode(f=self.evaluate_rhs,
                                              jac=self.evaluate_jacobian)
        return self.__integrator

    def _clear_cache(self):
        """Clear cached functions used for numerical evaluation."""
        super(ShootingSolver, self)._clear_cache()
        self.__numeric_jacobian = None
        self.__numeric_system = None
        self.__integrator = None

    def _converged_firms(self, tol):
        """Check whether solution component for firms has converged."""
        if abs(self.integrator.y[0] - self.model.firms.lower) <= tol:
            converged = True
        else:
            converged = False
        return converged

    def _converged_workers(self, tol):
        """Check whether solution component for workers has converged."""
        if self.model.assortativity == 'positive':
            if abs(self.integrator.t - self.model.workers.lower) <= tol:
                converged = True
            else:
                converged = False
        else:
            if abs(self.integrator.t - self.model.workers.upper) <= tol:
                converged = True
            else:
                converged = False

        return converged

    def _exhausted_firms(self, tol):
        """Check whether firms have been exhausted."""
        if self.integrator.y[0] - self.model.firms.lower < -tol:
            exhausted = True
        else:
            exhausted = False
        return exhausted

    def _guess_firm_size_upper_too_low(self, bound, tol):
        """Check whether guess for upper bound for firm size is too low."""
        return abs(self.integrator.y[1] - bound) <= tol

    def _reset_solution(self, firm_size):
        """
        Reset the initial condition for the integrator and re-initialze the
        solution array.

        Parameters
        ----------
        firm_size : float

        """
        x_lower, x_upper = self.model.workers.lower, self.model.workers.upper
        y_upper = self.model.firms.upper
        initial_V = np.array([y_upper, firm_size])

        if self.model.assortativity == 'positive':
            self.integrator.set_initial_value(initial_V, x_upper)
            wage = self.evaluate_wage(x_upper, initial_V)
            profit = self.evaluate_profit(x_upper, initial_V)
            self._solution = np.hstack((x_upper, initial_V, wage, profit))
        else:
            self.integrator.set_initial_value(initial_V, x_lower)
            wage = self.evaluate_wage(x_lower, initial_V)
            profit = self.evaluate_profit(x_lower, initial_V)
            self._solution = np.hstack((x_lower, initial_V, wage, profit))

    def _update_initial_guess(self, lower, upper):
        """
        Use bisection method to arrive at new initial guess for firm size.

        Parameters
        ----------
        lower : float
            Lower bound on the true initial condition for firm size.
        upper : float
            Upper bound on the true initial condition for firm size.

        Returns
        -------
        guess : float
            New initial guess for firm size.

        """
        err_mesg = 'Upper and lower bounds are identical: check solver tols!'
        assert (upper - lower) > np.finfo('float').eps, err_mesg
        guess = 0.5 * (lower + upper)
        return guess

    def _update_solution(self, step_size):
        """
        Update the solution array.

        Parameters
        ----------
        step_size : float
            Step size for determining next point in the solution.

        """
        if self.model.assortativity == 'positive':
            self.integrator.integrate(self.integrator.t - step_size)
            x, V = self.integrator.t, self.integrator.y
        else:
            self.integrator.integrate(self.integrator.t + step_size)
            x, V = self.integrator.t, self.integrator.y

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
        Numerically evaluate right-hand side of the system of ODEs.

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

    def solve(self, guess_firm_size_upper, tol=1e-6, number_knots=100,
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
        attribute of the `Solver` class. To final solution is stored as a
        `pandas.DataFrame` in the `solution` attribute.

        """

        # relevant bounds
        x_lower = self.model.workers.lower
        x_upper = self.model.workers.upper

        # initialize integrator
        self.integrator.set_integrator(integrator, **kwargs)

        # initialize the solution
        firm_size_lower = 0.0
        firm_size_upper = guess_firm_size_upper
        guess_firm_size = 0.5 * (firm_size_upper + firm_size_lower)
        self._reset_solution(guess_firm_size)

        # step size insures that never step beyond x_lower
        step_size = (x_upper - x_lower) / (number_knots - 1)
        assert step_size > 0

        while self.integrator.successful():

            if self._guess_firm_size_upper_too_low(guess_firm_size_upper, tol):
                if message:
                    mesg = ("Failure! Need to increase initial guess for " +
                            "upper bound on firm size!")
                    print(mesg)
                break

            self._update_solution(step_size)

            if self._converged_workers(tol) and self._converged_firms(tol):
                self._validate_solution(self._solution)
                mesg = "Success! All workers and firms are matched"
                print(mesg)
                break

            elif (not self._converged_workers(tol)) and self._exhausted_firms(tol):
                if message:
                    mesg = ("Exhausted firms: initial guess of {} for firm " +
                            "size is too low.")
                    print(mesg.format(guess_firm_size))
                firm_size_lower = guess_firm_size

            elif self._converged_workers(tol) and self._exhausted_firms(tol):
                if message:
                    mesg = ("Exhausted firms: Initial guess of {} for firm " +
                            "size was too low!")
                    print(mesg.format(guess_firm_size))
                firm_size_lower = guess_firm_size

            elif self._converged_workers(tol) and (not self._exhausted_firms(tol)):
                if message:
                    mesg = ("Exhausted workers: initial guess of {} for " +
                            "firm size is too high!")
                    print(mesg.format(guess_firm_size))
                firm_size_upper = guess_firm_size

            else:
                continue

            guess_firm_size = self._update_initial_guess(firm_size_lower,
                                                         firm_size_upper)
            self._reset_solution(guess_firm_size)
