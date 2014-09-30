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