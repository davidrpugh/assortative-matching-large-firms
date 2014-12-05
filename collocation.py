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
