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

    __nodes = None

    def __init__(self, model, kind="Chebyshev"):
        """Create an instance of the OrthogonalCollocation class."""
        super(OrthogonalCollocation, self).__init__(model)
        self.kind = kind

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
        valid_kinds = ['Chebyshev', 'Hermitian', 'Legendre', 'Laguerre']
        if not isinstance(kind, str):
            mesg = ("Attribute 'kind' must have type str, not {}.")
            raise AttributeError(mesg.format(kind.__class__))
        elif kind not in valid_kinds:
            mesg = "Attribute 'kind' must be one of {}."
            raise AttributeError(mesg.format(valid_kinds))
        else:
            return kind
