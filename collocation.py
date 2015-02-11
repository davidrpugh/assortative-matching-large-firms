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

    def __init__(self, model, kind="Chebyshev"):
        """Create an instance of the OrthogonalCollocation class."""
        super(OrthogonalCollocation, self).__init__(model)
        self.kind = kind

        # initialize coefficients to linear polynomials
        self.degree_mu, self.degree_theta = 1, 1
        self._coefficients_mu = self._initialize_coefficients_mu()
        self._coefficients_theta = self._intialize_coefficients_theta()

    @property
    def _boundary_conditions(self):
        """Boundary conditions for the problem."""
        if self.model.assortativity == 'positive':
            lower = (self.evaluate_mu(self.model.workers.lower) -
                     self.model.firms.lower)
            upper = (self.evaluate_mu(self.model.workers.upper) -
                     self.model.firms.upper)
        else:
            lower = (self.evaluate_mu(self.model.workers.lower) -
                     self.model.firms.upper)
            upper = (self.evaluate_mu(self.model.workers.upper) -
                     self.model.firms.lower)
        return np.hstack((lower, upper))

    @property
    def _coefficients_mu(self):
        return self.__coefficients_mu

    @_coefficients_mu.setter
    def _coefficients_mu(self, value):
        assert (value.size == self.degree_mu + 1)
        self.__coefficients_mu = value

    @property
    def _coefficients_theta(self):
        return self.__coefficients_theta

    @_coefficients_theta.setter
    def _coefficients_theta(self, value):
        assert (value.size == self.degree_theta + 1), "{}".format(value)
        self.__coefficients_theta = value

    @property
    def _collocation_nodes_mu(self):
        r"""Collocation nodes for approximation of :math:`\mu(x)`."""
        basis_coefs = np.zeros(self._coefficients_mu.size)
        basis_coefs[-1] = 1
        basis_poly = self.polynomial_factory(basis_coefs, self.kind)
        return basis_poly.roots()

    @property
    def _collocation_nodes_theta(self):
        r"""Collocation nodes for approximation of :math:`\theta(x)`."""
        basis_coefs = np.zeros(self._coefficients_theta.size)
        basis_coefs[-1] = 1
        basis_poly = self.polynomial_factory(basis_coefs, self.kind)
        return basis_poly.roots()

    @property
    def _collocation_system(self):
        """System of non-linear equations whose solution is the coefficients."""
        tup = (self.evaluate_residual_mu(self._collocation_nodes_mu),
               self.evaluate_residual_theta(self._collocation_nodes_theta),
               self._boundary_conditions,
               )
        return np.hstack(tup)

    @property
    def _domain(self):
        """Domain of approximation for the collocation solver."""
        return [self.model.workers.lower, self.model.workers.upper]

    @property
    def degree_mu(self):
        r"""
        Degree of the orthogonal polynomial used to approximate math:`\mu(x)`.

        :getter: Return the current degree.
        :setter: Set a new value for the degree.
        :type: int

        """
        return self._degree_mu

    @degree_mu.setter
    def degree_mu(self, degree):
        """Set a new value for the degree of approximating polynomial."""
        self._degree_mu = self._validate_degree(degree)

    @property
    def degree_theta(self):
        r"""
        Degree of the orthogonal polynomial used to approximate math:`\theta(x)`.

        :getter: Return the current degree.
        :setter: Set a new value for the degree.
        :type: int

        """
        return self._degree_theta

    @degree_theta.setter
    def degree_theta(self, degree):
        """Set a new value for the degree of approximating polynomial."""
        self._degree_theta = self._validate_degree(degree)

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
    def orthogonal_polynomial_mu(self):
        r"""
        Orthogonal polynomial approximation of the equilibrium assignment
        function, :math:`\mu(x)`.

        :getter: Return the orthogonal polynomial approximation.
        :type: numpy.polynomial.Polynomial

        """
        return self.polynomial_factory(self._coefficients_mu, self.kind)

    @property
    def orthogonal_polynomial_theta(self):
        r"""
        Orthogonal polynomial approximation of the firm size function,
        :math:`\theta(x)`.

        :getter: Return the orthogonal polynomial approximation.
        :type: numpy.polynomial.Polynomial

        """
        return self.polynomial_factory(self._coefficients_theta, self.kind)

    def _evaluate_collocation_residual(self, coefs):
        """Collocation residual should be zero for optimal coefficents."""
        self._coefficients_mu = coefs[:self.degree_mu+1]
        self._coefficients_theta = coefs[self.degree_mu+1:]
        return self._collocation_system

    def _initialize_coefficients_mu(self):
        """Intitialize the coefficients for the orthogonal polynomial mu."""
        # construct a basis polynomial of the appropriate kind
        basis_coefs = np.array([0.0, 1.0])
        tmp_polynomial = self.polynomial_factory(basis_coefs, self.kind)

        # fit a linear polynomial to get the inial coefs
        x = np.hstack((self.model.workers.lower, self.model.workers.upper))
        if self.model.assortativity == 'positive':
            y = np.hstack((self.model.firms.lower, self.model.firms.upper))
        else:
            y = np.hstack((self.model.firms.upper, self.model.firms.lower))
        linear_mu = tmp_polynomial.fit(x, y, 1)
        initial_coefs_mu = linear_mu.coef

        return initial_coefs_mu

    def _intialize_coefficients_theta(self):
        """Intitialize the coefficients for the orthogonal polynomial theta."""
        # construct a basis polynomial of the appropriate kind
        basis_coefs = np.array([0.0, 1.0])
        tmp_polynomial = self.polynomial_factory(basis_coefs, self.kind)

        # fit a linear polynomial to get the inial coefs
        x = np.hstack((self.model.workers.lower, self.model.workers.upper))
        slope = ((self.model.firms.upper - self.model.firms.lower) /
                 (self.model.workers.upper - self.model.workers.lower))
        if self.model.assortativity == 'positive':
            y = self.evaluate_density_ratio(x) / np.repeat(slope, 2)
        else:
            y = -self.evaluate_density_ratio(x) / np.repeat(slope, 2)
        linear_theta = tmp_polynomial.fit(x, y, 1)
        initial_coefs_theta = linear_theta.coef

        return initial_coefs_theta

    @staticmethod
    def _validate_degree(degree):
        """Validate the degree_mu and degree_theta attributes."""
        if not isinstance(degree, int):
            mesg = ("Attribute must have type int, not {}.")
            raise AttributeError(mesg.format(degree.__class__))
        elif degree <= 0:
            mesg = "Attribute must be strictly positive integer."
            raise AttributeError(mesg)
        else:
            return degree

    @staticmethod
    def _validate_kind(kind):
        """Validate the kind attribute."""
        valid_kinds = ['Chebyshev', 'Hermite', 'Legendre', 'Laguerre']
        if not isinstance(kind, str):
            mesg = ("Attribute 'kind' must have type str, not {}.")
            raise AttributeError(mesg.format(kind.__class__))
        elif kind not in valid_kinds:
            mesg = "Attribute 'kind' must be one of {}."
            raise AttributeError(mesg.format(valid_kinds))
        else:
            return kind

    def evaluate_mu(self, x):
        r"""Numerically evaluate the solution function :math:`\hat{\mu}(x)`."""
        return self.orthogonal_polynomial_mu(x)

    def evaluate_mu_prime(self, x):
        r"""Numerically evaluate the solution function :math:`\hat{\mu}'(x)`."""
        theta_prime = self.orthogonal_polynomial_mu.deriv()
        return theta_prime(x)

    def evaluate_theta(self, x):
        r"""Numerically evaluate the solution function :math:`\hat{\theta}(x)`."""
        return self.orthogonal_polynomial_theta(x)

    def evaluate_theta_prime(self, x):
        r"""Numerically evaluate the solution function :math:`\hat{\theta}'(x)`."""
        theta_prime = self.orthogonal_polynomial_theta.deriv()
        return theta_prime(x)

    def polynomial_factory(self, coefficients, kind):
        """
        Factory method for generating various orthogonal polynomials.

        Parameters
        ----------
        coefficients : numpy.ndarray (shape=(N,))
            Array of polynomial coefficients.
        kind : string
            Class of orthogonal polynomials to use as basic functions. Must be
            one of "Chebyshev", "Hermite", "Laguerre", or "Legendre."

        Returns
        -------
        polynomial : numpy.polynomial.Polynomial
            Approximating polynomial.

        """
        if kind == "Chebyshev":
            polynomial = np.polynomial.Chebyshev(coefficients, self._domain)
        elif kind == "Hermite":
            polynomial = np.polynomial.Hermite(coefficients, self._domain)
        elif kind == "Laguerre":
            polynomial = np.polynomial.Laguerre(coefficients, self._domain)
        elif kind == "Legendre":
            polynomial = np.polynomial.Legendre(coefficients, self._domain)
        else:
            mesg = ("Somehow you managed to specify an invalid 'kind' of " +
                    "orthogonal polynomials!")
            raise ValueError(mesg)
        return polynomial

    def solve(self, initial_coefs, method='hybr', **kwargs):
        """Solve the system of non-linear equations."""
        result = optimize.root(self._evaluate_collocation_residual,
                               x0=initial_coefs,
                               method=method,
                               **kwargs)
        return result
