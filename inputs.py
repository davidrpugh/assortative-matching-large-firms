from __future__ import division

import sympy as sym


class Input(object):
    """Class representing a heterogenous production input."""

    def __init__(self, var, cdf, bounds, params):
        """
        Create an instance of the Input class.

        Parameters
        ----------
        var : sym.Symbol
            Symbolic variable representing the production input.
        cdf : sym.Basic
            Symbolic expression defining a valid probability distribution
            function (CDF). Must be a function of var.
        bounds : list
            List containing the lower and upper bounds on the support of the
            probability distribution function (CDF).
        params : dict
            Dictionary of distribution parameters.

        """
        self.var = var
        self.cdf = cdf
        self.lower = bounds[0]
        self.upper = bounds[1]
        self.params = params

    @property
    def cdf(self):
        """
        Probability distribution function (CDF).

        :getter: Return the current distribution function.
        :setter: Set a new distribution function.
        :type: sym.Basic

        """
        return self._cdf

    @cdf.setter
    def cdf(self, value):
        """Set a new probability distribution function (CDF)."""
        self._cdf = self._validate_cdf(value)

    @property
    def lower(self):
        """
        Lower bound on support of the probability distribution function (CDF).

        :getter: Return the lower bound.
        :setter: Set a new lower bound.
        :type: float

        """
        return self._lower

    @lower.setter
    def lower(self, value):
        """Set a new lower bound."""
        self._lower = self._validate_lower_bound(value)

    @property
    def params(self):
        """
        Dictionary of distribution parameters.

        :getter: Return the current parameter dictionary.
        :setter: Set a new parameter dictionary.
        :type: dict

        """
        return self._params

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        self._params = self._validate_params(value)

    @property
    def upper(self):
        """
        Upper bound on support of the probability distribution function (CDF).

        :getter: Return the lower bound.
        :setter: Set a new lower bound.
        :type: float

        """
        return self._upper

    @upper.setter
    def upper(self, value):
        """Set a new upper bound."""
        self._upper = self._validate_upper_bound(value)

    @property
    def var(self):
        """
        Symbolic variable respresenting the production input.

        :getter: Return the current variable.
        :setter: Set a new variable.
        :type: sym.Symbol

        """
        return self._var

    @var.setter
    def var(self, value):
        """Set a new symbolic variable."""
        self._var = self._validate_var(value)

    @staticmethod
    def _validate_cdf(cdf):
        """Validates the probability distribution function (CDF)."""
        if not isinstance(cdf, sym.Basic):
            mesg = "Attribute 'cdf' must have type sympy.Basic, not {}"
            raise AttributeError(mesg.format(cdf.__class__))
        else:
            return cdf

    def _validate_lower_bound(self, value):
        """Validate the lower bound on the suppport of the CDF."""
        if not isinstance(value, float):
            mesg = "Attribute 'lower' must have type float, not {}"
            raise AttributeError(mesg.format(value.__class__))
        elif value > self.upper:
            mesg = "Lower bound must be less than the upper bound!"
            raise AttributeError(mesg)
        else:
            return value

    @staticmethod
    def _validate_params(value):
        """Validate the dictionary of parameters."""
        if not isinstance(value, dict):
            mesg = "Attribute 'params' must have type dict, not {}"
            raise AttributeError(mesg.format(value.__class__))
        else:
            return value

    def _validate_upper_bound(self, value):
        """Validate the upper bound on the suppport of the CDF."""
        if not isinstance(value, float):
            mesg = "Attribute 'upper' must have type float, not {}"
            raise AttributeError(mesg.format(value.__class__))
        elif value < self.lower:
            mesg = "Upper bound must be greater than the lower bound!"
            raise AttributeError
        else:
            return value

    @staticmethod
    def _validate_var(var):
        """Validates the symbolic variable."""
        if not isinstance(var, sym.Symbol):
            mesg = "Attribute 'var' must have type sympy.Symbol, not {}"
            raise AttributeError(mesg.format(var.__class__))
        else:
            return var