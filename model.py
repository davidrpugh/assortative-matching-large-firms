import sympy as sym

# define symbolic variables
l, r, x, y = sym.var('l, r, x, y')


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
    def Flr(self):
        return sym.diff(self.F, l, r)

    @property
    def Fxr(self):
        return sym.diff(self.F, x, r)

    @property
    def Fyl(self):
        return sym.diff(self.F, y, l)

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
        else:
            return params

    @staticmethod
    def _validate_production_function(F):
        """Validates the production function attribute."""
        if not isinstance(F, sym.Basic):
            mesg = "Attribute 'F' must have type sympy.Basic, not {}."
            raise AttributeError(mesg.format(F.__class__))
        elif not {l, r} < F.atoms():
            mesg = "Attribute 'F' must be an expression of r and l."
            raise AttributeError(mesg)
        else:
            return F


class DifferentiableMatching(object):
    """Base class representing a differentiable matching system of ODEs."""

    def __init__(self, model):
        """
        Create an instance of the DifferentiableMatching class.

        Parameters
        model : model.Model
            Instance of the model.Model class representing a matching model
            with two-sided heterogeneity.

        """
        self.model = model

    @property
    def H(self):
        raise NotImplementedError

    @property
    def model(self):
        """
        Instance of the model.Model class representing a matching model
        with two-sided heterogeneity.

        :getter: Return the current model.Model instance.
        :setter: Set a new model.Model instance
        :type: model.Model

        """
        return self._model

    @model.setter
    def model(self, model):
        """Set a new model.Model instance."""
        self._model = self._validate_model(model)

    @property
    def mu_prime(self):
        raise NotImplementedError

    @property
    def theta_prime(self):
        raise NotImplementedError

    @staticmethod
    def _validate_model(model):
        """Validates the model attribute."""
        if not isinstance(model, Model):
            mesg = "Attribute 'model' must have type model.Model, not {}."
            raise AttributeError(mesg.format(model.__class__))
        else:
            return model
