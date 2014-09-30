from __future__ import division

import sympy as sym


class Input(object):
    """Class representing a heterogenous production input."""

    def __init__(self, bounds, distribution, params):
        """
        Create an instance of the Input class.

        Parameters
        ----------
        bounds : list
            List containing the lower and uppoer bounds on the support of the
            probability distribution.
        distribution : sym.Basic
            Symbolic expression defining a valid probability distribution
            function.
        params : dict
            Dictionary of distribution parameters.

        """
        self.lower = bounds[0]
        self.upper = bounds[1]
        self.distribution = distribution
        self.params = params

    @property
    def distribution(self):
        """
        Probability distribution function (CDF).

        :getter: Return the current distribution function.
        :setter: Set a new distribution function.
        :type: sym.Basic

        """
        return self._distribution

    @distribution.setter
    def distribution(self, value):
        """Set a new probability distribution function."""
        self._distribution = self._validate_distribution(value)

    @staticmethod
    def _validate_distribution(cdf):
        """Validates the probability distribution function."""
        if not isinstance(cdf, sym.Basic):
            raise AttributeError
        else:
            return cdf
