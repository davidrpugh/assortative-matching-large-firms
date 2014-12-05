"""
Contains the base class for the solver.py module.

@author : David R. Pugh

"""
import numpy as np
import pandas as pd
from scipy import special
import sympy as sym

import models

# represent endogenous variables mu and theta as a deferred vector
V = sym.DeferredVector('V')


class Solver(object):
    """Base class for all solvers."""

    __numeric_input_types = None

    __numeric_profit = None

    __numeric_quantities = None

    __numeric_span_of_control = None

    __numeric_type_resource = None

    __numeric_wage = None

    _modules = [{'ImmutableMatrix': np.array, 'erf': special.erf}, 'numpy']

    def __init__(self, model):
        """
        Create an instance of the Solver class.

        """
        self.model = model

    @property
    def _numeric_input_types(self):
        """
        Vectorized function for numerical evaluation of the input type
        complementarity.

        :getter: Return current function for evaluating the complementarity.
        :type: function

        """
        if self.__numeric_input_types is None:
            self.__numeric_input_types = sym.lambdify(self._symbolic_args,
                                                      self._symbolic_input_types,
                                                      self._modules)
        return self.__numeric_input_types

    @property
    def _numeric_profit(self):
        """
        Vectorized function for numerical evaluation of profits.

        :getter: Return current function for evaluating profits.
        :type: function

        """
        if self.__numeric_profit is None:
            self.__numeric_profit = sym.lambdify(self._symbolic_args,
                                                 self._symbolic_profit,
                                                 self._modules)
        return self.__numeric_profit

    @property
    def _numeric_quantities(self):
        """
        Vectorized function for numerical evaluation of the quantity
        complementarity.

        :getter: Return current function for evaluating the complementarity.
        :type: function

        """
        if self.__numeric_quantities is None:
            self.__numeric_quantities = sym.lambdify(self._symbolic_args,
                                                     self._symbolic_quantities,
                                                     self._modules)
        return self.__numeric_quantities

    @property
    def _numeric_span_of_control(self):
        """
        Vectorized function for numerical evaluation of the resource
        complementarity.

        :getter: Return current function for evaluating the complementarity.
        :type: function

        """
        if self.__numeric_span_of_control is None:
            self.__numeric_span_of_control = sym.lambdify(self._symbolic_args,
                                                          self._symbolic_span_of_control,
                                                          self._modules)
        return self.__numeric_span_of_control

    @property
    def _numeric_type_resource(self):
        """
        Vectorized function for numerical evaluation of the resource
        complementarity.

        :getter: Return current function for evaluating the complementarity.
        :type: function

        """
        if self.__numeric_type_resource is None:
            self.__numeric_type_resource = sym.lambdify(self._symbolic_args,
                                                        self._symbolic_type_resource,
                                                        self._modules)
        return self.__numeric_type_resource

    @property
    def _numeric_wage(self):
        """
        Vectorized function for numerical evaluation of wages.

        :getter: Return current function for evaluating wages.
        :type: function

        """
        if self.__numeric_wage is None:
            self.__numeric_wage = sym.lambdify(self._symbolic_args,
                                               self._symbolic_wage,
                                               self._modules)
        return self.__numeric_wage

    @property
    def _solution(self):
        """
        Solution to the model represented as a NumPy array.

        :getter: Return the array represnting the current solution
        :setter: Set a new array defining the solution.
        :type: numpy.ndarray

        """
        return self.__solution

    @_solution.setter
    def _solution(self, value):
        """Set a new value for the solution array."""
        self.__solution = value

    @property
    def _symbolic_input_types(self):
        """
        Symbolic expression for complementarity between input types.

        :getter: Return the current expression for the complementarity.
        :type: sympy.Basic

        """
        Fxy = self.model.matching.input_types
        return Fxy.subs({'mu': V[0], 'theta': V[1]})

    @property
    def _symbolic_profit(self):
        """
        Symbolic expression defining profit.

        :getter: Return the symbolic expression for profits.
        :type: sympy.Basic

        """
        profit = self.model.matching.profit
        return profit.subs({'mu': V[0], 'theta': V[1]})

    @property
    def _symbolic_quantities(self):
        """
        Symbolic expression for complementarity between input quantities.

        :getter: Return the current expression for the complementarity.
        :type: sympy.Basic

        """
        Flr = self.model.matching.quantities
        return Flr.subs({'mu': V[0], 'theta': V[1]})

    @property
    def _symbolic_span_of_control(self):
        """
        Symbolic expression for span-of-control complementarity.

        :getter: Return the current expression for the complementarity.
        :type: sympy.Basic

        """
        Fyl = self.model.matching.span_of_control
        return Fyl.subs({'mu': V[0], 'theta': V[1]})

    @property
    def _symbolic_type_resource(self):
        """
        Symbolic expression for complementarity between worker type and
        firm resources.

        :getter: Return the current expression for the complementarity.
        :type: sympy.Basic

        """
        Fxr = self.model.matching.type_resource
        return Fxr.subs({'mu': V[0], 'theta': V[1]})

    @property
    def _symbolic_wage(self):
        """
        Symbolic expression defining wages.

        :getter: Return the symbolic expression for wages.
        :type: sympy.Basic

        """
        wage = self.model.matching.wage
        return wage.subs({'mu': V[0], 'theta': V[1]})

    @property
    def model(self):
        """
        Instance of the models.Model class to be solved via forward shooting.

        :getter: Return the current models.Model instance.
        :setter: Set a new models.Model instance.
        :type: models.Model

        """
        return self._model

    @model.setter
    def model(self, model):
        """Set a new Model attribute."""
        self._model = self._validate_model(model)
        self._clear_cache()

    @property
    def solution(self):
        """
        Solution to the model represented as a Pandas DataFrame.

        :getter: Return the DataFrame representing the current solution.
        :type: pandas.DataFrame

        """
        col_names = ['x', 'firm productivity', 'firm size', 'wage', 'profit']
        df = pd.DataFrame(self._solution, columns=col_names)
        if self.model.assortativity == 'positive':
            df.sort('x', inplace=True)
        else:
            pass
        return df.set_index('x')

    def _check_pam(self, step):
        r"""
        Check necessary condition required for a positive assortative
        matching (PAM).

        Parameters
        ----------
        step : numpy.ndarray (shape=(5,))
            Step along a putative solution to the model.

        Returns
        -------
        check : boolean
            Flag indicating whether positive assortative matching condition is
            satisfied for the given step.

        """
        # unpack the step
        x, V = step[0], step[1:3]

        LHS = self.evaluate_input_types(x, V) * self.evaluate_quantities(x, V)
        RHS = (self.evaluate_span_of_control(x, V) *
               self.evaluate_type_resource(x, V))

        if np.isclose(LHS - RHS, 0):
            check = True
        else:
            check = LHS > RHS

        return check

    def _clear_cache(self):
        """Clear cached functions used for numerical evaluation."""
        self.__numeric_input_types = None
        self.__numeric_profit = None
        self.__numeric_quantities = None
        self.__numeric_span_of_control = None
        self.__numeric_type_resource = None
        self.__numeric_wage = None

    @staticmethod
    def _validate_model(model):
        """Validate the model attribute."""
        if not isinstance(model, models.Model):
            mesg = ("Attribute 'model' must have type models.Model, not {}.")
            raise AttributeError(mesg.format(model.__class__))
        else:
            return model

    def _validate_solution(self, solution):
        """Validate a putative solution to the model."""
        check = np.apply_along_axis(self._check_pam, axis=1, arr=solution)
        if self.model.assortativity == 'positive' and (not check.all()):
            mesg = ("Approximated solution failed to satisfy required " +
                    "assortativity condition.")
            raise ValueError(mesg)
        elif self.model.assortativity == 'negative' and (check.all()):
            mesg = ("Approximated solution failed to satisfy required " +
                    "assortativity condition.")
            raise ValueError(mesg)
        else:
            pass

    def evaluate_input_types(self, x, V):
        r"""
        Numerically evaluate complementarity between input types.

        Parameters
        ----------
        x : float
            Value for worker skill (i.e., the independent variable).
        V : numpy.array (shape=(2,))
            Array of values for the dependent variables with ordering:
            :math:`[\mu, \theta]`.

        Returns
        -------
        input_types : float
            Complementarity between input types.

        """
        input_types = self._numeric_input_types(x, V, *self.model.params.values())
        return input_types

    def evaluate_profit(self, x, V):
        r"""
        Numerically evaluate profit for a firm with productivity V[0] and size
        V[1] when matched with a worker with skill x.

        Parameters
        ----------
        x : float
            Value for worker skill (i.e., the independent variable).
        V : numpy.array (shape=(2,))
            Array of values for the dependent variables with ordering:
            :math:`[\mu, \theta]`.

        Returns
        -------
        profit : float
            Firm's profit.

        """
        profit = self._numeric_profit(x, V, *self.model.params.values())
        assert profit > 0.0, "Profit should be non-negative!"
        return profit

    def evaluate_quantities(self, x, V):
        r"""
        Numerically evaluate quantities complementarity.

        Parameters
        ----------
        x : float
            Value for worker skill (i.e., the independent variable).
        V : numpy.array (shape=(2,))
            Array of values for the dependent variables with ordering:
            :math:`[\mu, \theta]`.

        Returns
        -------
        quantities : float
            Complementarity between quantities

        """
        quantities = self._numeric_quantities(x, V, *self.model.params.values())
        return quantities

    def evaluate_rhs_mu(self, x, V):
        raise NotImplementedError

    def evaluate_rhs_theta(self, x, V):
        raise NotImplementedError

    def evaluate_type_resource(self, x, V):
        r"""
        Numerically evaluate complementarity between worker skill and
        firm resources.

        Parameters
        ----------
        x : float
            Value for worker skill (i.e., the independent variable).
        V : numpy.array (shape=(2,))
            Array of values for the dependent variables with ordering:
            :math:`[\mu, \theta]`.

        Returns
        -------
        resource : float
            Complementarity between worker skill and firm resources.

        """
        resource = self._numeric_type_resource(x, V, *self.model.params.values())
        return resource

    def evaluate_span_of_control(self, x, V):
        r"""
        Numerically evaluate span-of-control complementarity.

        Parameters
        ----------
        x : float
            Value for worker skill (i.e., the independent variable).
        V : numpy.array (shape=(2,))
            Array of values for the dependent variables with ordering:
            :math:`[\mu, \theta]`.

        Returns
        -------
        span_of_control : float
            Span-of-control complementarity.

        """
        span_of_control = self._numeric_span_of_control(x, V, *self.model.params.values())
        return span_of_control

    def evaluate_wage(self, x, V):
        r"""
        Numerically evaluate wage for a worker with skill level x when matched
        to a firm with productivity V[0] with size V[1].

        Parameters
        ----------
        x : float
            Value for worker skill (i.e., the independent variable).
        V : numpy.array (shape=(2,))
            Array of values for the dependent variables with ordering:
            :math:`[\mu, \theta]`.

        Returns
        -------
        wage : float
            Worker's wage.

        """
        wage = self._numeric_wage(x, V, *self.model.params.values())
        assert wage > 0.0, "Wage should be non-negative!"
        return wage

    def solve(self, *args, **kwargs):
        raise NotImplementedError
