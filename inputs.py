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
        self.distribution = distribution
        self.lower = bounds[0]
        self.upper = bounds[1]
        self.params = params
