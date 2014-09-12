"""

TODO:

    1) Derive an expression for normalized density function that takes into account
       the measure of available input.
    2) Find a way to check multiple SymPy types using Enum trait.
    3) Need to deal with circular dependencies in Properties.
    4) Need to automatically detect atoms used in symbolic cdf/pdf.
    5) Write some tests!

"""
from __future__ import division

import sympy as sp 

from traits.api import (cached_property, Any, Dict, Float, HasPrivateTraits, 
                        Property, Str)

class Input(HasPrivateTraits):
    """Class representing a heterogenous production input."""

    _symbolic_cdf = Property(depends_on=['distribution', 'params'])

    _symbolic_pdf = Property(depends_on='_symbolic_cdf')

    # why does this cause an error if depend_on is specified?
    numeric_normalized_pdf = Property

    symbolic_normalized_pdf = Property(depends_on=['_symbolic_pdf', 
                                                   'lower_bound', 
                                                   'upper_bound'])

    # ideally this should be one of the core sympy classes!
    distribution = Any

    lower_bound = Float

    params = Dict(Str, Float)

    upper_bound = Float

    @cached_property
    def _get__symbolic_cdf(self):
        """Symbolic expression for the CDF of input heterogeneity."""
        return self.distribution.subs(self.params)

    @cached_property
    def _get__symbolic_pdf(self):
        """Symbolic expression for normalized pdf of input heterogeneity."""
        sp.var('x')
        return sp.diff(self._symbolic_cdf, x)

    @cached_property
    def _get_numeric_normalized_pdf(self):
        """Numeric function for normalized pdf of input heterogeneity."""
        arg = sp.var('x')
        return sp.lambdify(arg, self.symbolic_normalized_pdf, modules='numpy')

    @cached_property
    def _get_symbolic_normalized_pdf(self):
        """Symbolic expression for normalized pdf of input heterogeneity."""
        norm_constant = (self._symbolic_cdf.evalf(subs={'x':self.upper_bound}) - 
                         self._symbolic_cdf.evalf(subs={'x':self.lower_bound}))

        return self._symbolic_pdf / norm_constant


if __name__ == '__main__':
    import sympy as sp
    
    # suppose worker skill is log normal
    sp.var('x, mu, sigma')
    skill_cdf = 0.5 + 0.5 * sp.erf((sp.log(x) - mu) / sp.sqrt(2 * sigma**2))
    worker_params = {'mu':0.0, 'sigma':1}
    x_lower, x_upper = 1e-3, 5e1
    
    workers = Input(distribution=skill_cdf,
                    lower_bound=x_lower,
                    params=worker_params,
                    upper_bound=x_upper,
                   )

    print(workers.symbolic_normalized_pdf)


