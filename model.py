import sympy as sym


class Model(object):
    """Class representing a matching model with two-sided heterogeneity."""

    def __init__(self, production_function, params):
        """
        Create an instance of the Model class.

        Parameters
        ----------
        production_function : sym.Basic
            Symbolic expression describing the production technology.
        params : dict
            Dictionary of model parameters.

        """
        self.F = production_function
        self.params = params

    @property
    def F(self):
        """
        Symbolic expression describing the available production technology.

        :getter: Return the current production function.
        :setter: Set a new production function.
        :type: sym.Basic

        """
        return self._F

    @F.setter
    def F(self, value):
        """Set a new production function."""
        self._F = self._validate_production_function(value)

    @property
    def params(self):
        """
        Dictionary of model parameters.

        :getter: Return the current parameter dictionary.
        :setter: Set a new parameter dictionary.
        :type: dict

        """
        return self._params

    @params.setter
    def params(self, value):
        """Set a new parameter dictionary."""
        self._params = self._validate_params(value)

    @staticmethod
    def _validate_params(params):
        """Validates the dictionary of model parameters."""
        if not isinstance(params, dict):
            mesg = "Attribute 'params' must have type dict, not {}."
            raise AttributeError(mesg.format(params.__class__))

    @staticmethod
    def _validate_production_function(F):
        """Validates the production function attribute."""
        if not isinstance(F, sym.Basic):
            mesg = "Attribute 'F' must have type sympy.Basic, not {}."
            raise AttributeError(mesg.format(F.__class__))
    