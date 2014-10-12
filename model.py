import sympy as sym

import inputs

# define symbolic variables
l, r = sym.var('l, r')

# should really use letters for variables so as not to confound with params!
mu, theta = sym.var('mu, theta')


class Model(object):
    """Class representing a matching model with two-sided heterogeneity."""

    def __init__(self, assortativity, workers, firms, production, params):
        """
        Create an instance of the Model class.

        Parameters
        ----------
        assortativity : str
            String defining the type of matching assortativity. Must be one of
            'positive' or 'negative'.
        workers : inputs.Input
            Instance of the inputs.Input class defining workers with
            heterogeneous skill levels.
        firms : inputs.Input
            Instance of the inputs.Input class defining firms with
            heterogeneous productivity.
        production : sympy.Basic
            Symbolic expression describing the production technology.
        params : dict
            Dictionary of model parameters.

        """
        self.assortativity = assortativity
        self.workers = workers
        self.firms = firms
        self.F = production
        self.params = params

    @property
    def assortativity(self):
        """
        String defining the matching assortativty.

        :getter: Return the current matching assortativity
        :setter: Set a new matching assortativity.
        :type: str

        """
        return self._assortativity

    @assortativity.setter
    def assortativity(self, value):
        """Set new matching assortativity."""
        self._assortativity = self._validate_assortativity(value)

    @property
    def F(self):
        """
        Symbolic expression describing the available production technology.

        :getter: Return the current production function.
        :setter: Set a new production function.
        :type: sympy.Basic

        """
        return self._F

    @F.setter
    def F(self, value):
        """Set a new production function."""
        self._F = self._validate_production_function(value)

    @property
    def firms(self):
        """
        Instance of the inputs.Input class describing firms with heterogeneous
        productivity.

        :getter: Return current firms.
        :setter: Set new firms.
        :type: inputs.Input

        """
        return self._firms

    @firms.setter
    def firms(self, value):
        """Set new firms."""
        self._firms = self._validate_input(value)

    @property
    def Fxy(self):
        """
        Symbolic expression for the skill complementarity.

        :getter: Return the expression for the skill complementarity.
        :type: sympy.Basic

        """
        return sym.diff(self.F, self.workers.var, self.firms.var)

    @property
    def Flr(self):
        """
        Symbolic expression for the quantities complementarity.

        :getter: Return the expression for the quantities complementarity.
        :type: sympy.Basic

        """
        return sym.diff(self.F, l, r)

    @property
    def Fxr(self):
        """
        Symbolic expression for the managerial resource complementarity.

        :getter: Return the expression for managerial resource complementarity.
        :type: sympy.Basic

        """
        return sym.diff(self.F, self.workers.var, r)

    @property
    def Fyl(self):
        """
        Symbolic expression for the span-of-control complementarity.

        :getter: Return the expression for the span-of-control complementarity.
        :type: sympy.Basic

        """
        return sym.diff(self.F, self.firms.var, l)

    @property
    def matching(self):
        """
        Instance of the DifferentiableMatching class describing the matching
        equilibrium.

        :getter: Return the current DifferentiableMatching instance.
        :type: DifferentiableMatching

        """
        if self.assortativity == 'positive':
            return PositiveAssortativeMatching(self)
        else:
            return NegativeAssortativeMatching(self)

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

    @property
    def workers(self):
        """
        Instance of the inputs.Input class describing workers with
        heterogeneous skill.

        :getter: Return current workers.
        :setter: Set new workers.
        :type: inputs.Input

        """
        return self._workers

    @workers.setter
    def workers(self, value):
        """Set new workers."""
        self._workers = self._validate_input(value)

    @staticmethod
    def _validate_assortativity(value):
        """Validates the matching assortativity."""
        valid_assortativities = ['positive', 'negative']
        if not isinstance(value, str):
            mesg = "Attribute 'assortativity' must have type str, not {}."
            raise AttributeError(mesg.format(value.__class__))
        elif value not in valid_assortativities:
            mesg = "Attribute 'assortativity' must be in {}."
            raise AttributeError(mesg.format(valid_assortativities))
        else:
            return value

    @staticmethod
    def _validate_input(value):
        """Validates the worker and firm attributes."""
        if not isinstance(value, inputs.Input):
            mesg = ("Attributes 'workers' and 'firms' must have " +
                    "type inputs.Input, not {}.")
            raise AttributeError(mesg.format(value.__class__))
        else:
            return value

    @staticmethod
    def _validate_params(params):
        """Validates the dictionary of model parameters."""
        if not isinstance(params, dict):
            mesg = "Attribute 'params' must have type dict, not {}."
            raise AttributeError(mesg.format(params.__class__))
        else:
            return params

    def _validate_production_function(self, F):
        """Validates the production function attribute."""
        if not isinstance(F, sym.Basic):
            mesg = "Attribute 'F' must have type sympy.Basic, not {}."
            raise AttributeError(mesg.format(F.__class__))
        elif not {l, r} < F.atoms():
            mesg = "Attribute 'F' must be an expression of r and l."
            raise AttributeError(mesg)
        elif not {self.workers.var, self.firms.var} < F.atoms():
            mesg = ("Attribute 'F' must be an expression of workers.var and " +
                    "firm.var variables.")
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
    def _subs(self):
        """
        Dictionary of variable substitutions

        :getter: Return the current dictionary of substitutions.
        :type: dict

        """
        return {self.model.firms.var: mu, l: theta, r: 1.0}

    @property
    def H(self):
        """
        Ratio of worker probability density to firm probability density.

        :getter: Return current density ratio.
        :type: sympy.Basic

        """
        return self.model.workers.pdf / self.model.firms.pdf

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


class NegativeAssortativeMatching(DifferentiableMatching):
    """Class representing a model with negative assortative matching."""

    @property
    def mu_prime(self):
        expr = -self.H / theta
        return expr.subs(self._subs)

    @property
    def theta_prime(self):
        expr = -(self.H * self.model.Fyl + self.model.Fxr) / self.model.Flr
        return expr.subs(self._subs)


class PositiveAssortativeMatching(DifferentiableMatching):
    """Class representing a model with positive assortative matching."""

    @property
    def mu_prime(self):
        expr = self.H / theta
        return expr.subs(self._subs)

    @property
    def theta_prime(self):
        expr = (self.H * self.model.Fyl - self.model.Fxr) / self.model.Flr
        return expr.subs(self._subs)
