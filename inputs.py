"""
Classes for modeling heterogenous factors of production.

@author : David R. Pugh
@date : 2014-09-30

"""
from __future__ import division

import numpy as np
import sympy as sym


class Input(object):
    """Class representing a heterogenous production input."""

    modules = [{'ImmutableMatrix': np.array}, "numpy", "math"]

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

        # initialice cached values
        self.__numeric_cdf = None
        self.__numeric_pdf = None

    @property
    def _numeric_cdf(self):
        """
        Vectorized function used to numerically evaluate the CDF.

        :getter: Return the lambdified CDF.
        :type: function

        """
        if self.__numeric_cdf is None:
            args = [self.var] + sym.var(list(self.params.keys()))
            self.__numeric_cdf = sym.lambdify(args, self.cdf, self.modules)
        return self.__numeric_cdf

    @property
    def _numeric_pdf(self):
        """
        Vectorized function used to numerically evaluate the pdf.

        :getter: Return the lambdified pdf.
        :type: function

        """
        if self.__numeric_pdf is None:
            args = [self.var] + sym.var(list(self.params.keys()))
            self.__numeric_pdf = sym.lambdify(args, self.pdf, self.modules)
        return self.__numeric_pdf

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
    def norm_constant(self):
        """
        Constant used to normalize the probability density function (pdf).

        :getter: Return the current normalization constant.
        :type: float

        """
        return self.evaluate_cdf(self.upper) - self.evaluate_cdf(self.lower)

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
    def pdf(self):
        """
        Probability density function (pdf).

        :getter: Return the current probability density function.
        :type: sym.Basic

        """
        return sym.diff(self.cdf, self.var)

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

    def evaluate_cdf(self, value):
        """
        Numerically evaluate the probability distribution function (CDF).

        Parameters
        ----------
        value : numpy.ndarray
            Values at which to evaluate the CDF.

        Returns
        -------
        out : numpy.ndarray
            Evaluated CDF.

        """
        out = self._numeric_cdf(value, **self.params)
        return out

    def evaluate_pdf(self, value, norm=True):
        """
        Numerically evaluate the probability density function (pdf).

        Parameters
        ----------
        value : numpy.ndarray
            Values at which to evaluate the pdf.
        norm : boolean
            True if you wish to normalize the pdf so that it integrates to one;
            false otherwise.

        Returns
        -------
        out : numpy.ndarray
            Evaluated pdf.

        """
        if norm:
            out = self._numeric_pdf(value, **self.params) / self.norm_constant
        else:
            out = self._numeric_pdf(value, **self.params)
        return out
