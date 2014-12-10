"""
Module implementing an orthogonal collocation solver.

@author : David R. Pugh
@date : 2014-12-05

"""
from __future__ import division

import numpy as np

import solvers


class OrthogonalCollocation(solvers.Solver):
    """Class representing an orthogonal collocation solver."""

    def __init__(self, model, kind="Chebyshev"):
        """Create an instance of the OrthogonalCollocation class."""
        super(OrthogonalCollocation, self).__init__(model)
        self.kind = kind

        # initialize coefficients to linear polynomials
        self._coefficients_mu = np.array([0.0, 1.0])
        self._coefficients_theta = np.array([0.0, 1.0])

    @property
    def _boundary_conditions(self):
        """Boundary conditions for the problem."""
        if self.model.matching.assortativity == 'positive':
            lower = (self.orthogonal_polynomial_mu(self.model.workers.lower) -
                     self.model.firms.lower)
            upper = (self.orthogonal_polynomial_mu(self.model.workers.upper) -
                     self.model.firms.upper)
        else:
            lower = (self.orthogonal_polynomial_mu(self.model.workers.lower) -
                     self.model.firms.upper)
            upper = (self.orthogonal_polynomial_mu(self.model.workers.upper) -
                     self.model.firms.lower)
        return np.hstack((lower, upper))

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
        tup = (self._boundary_conditions,
               self.evaluate_residual_mu(self._collocation_nodes_mu),
               self.evaluate_residual_theta(self._collocation_nodes_theta))
        return np.hstack(tup)

    @property
    def _domain(self):
        """Domain of approximation for the collocation solver."""
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

    def evaluate_boundary_conditions(self, x):
        raise NotImplementedError

    def evaluate_collocation_system(self, x):
        raise NotImplementedError

    def evaluate_residual_mu(self, x):
        r"""
        Numerically evaluate the residual function for the orthogonal
        polynomial approximation of :math:`\mu(x)`

        Parameters
        ----------
        x : numpy.
            Value for worker skill (i.e., the independent variable).

        Returns
        -------
        residual : numpy.ndarray
            The residual function for the orthogonal polynomial approximation.

        """
        V = np.hstack((self.orthogonal_polynomial_mu(x),
                       self.orthogonal_polynomial_theta(x)))
        residual = (self.orthogonal_polynomial_mu.deriv()(x) -
                    self.evaluate_rhs_mu(x, V))
        return residual

    def evaluate_residual_theta(self, x):
        r"""
        Numerically evaluate the residual function for the polynomial
        approximation of :math:`\theta(x)`

        Parameters
        ----------
        x : numpy.
            Value for worker skill (i.e., the independent variable).

        Returns
        -------
        residual : numpy.ndarray
            The residual function for the orthogonal polynomial approximation.

        """
        V = np.hstack((self.orthogonal_polynomial_mu(x),
                       self.orthogonal_polynomial_theta(x)))
        residual = (self.orthogonal_polynomial_theta.deriv()(x) -
                    self.evaluate_rhs_theta(x, V))
        return residual

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
