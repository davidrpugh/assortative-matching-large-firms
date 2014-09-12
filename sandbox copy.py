"""

TODO:
    
    * Write some tests.
    * Start a private repo on github for this project.
    * Try to get a better understanding of how Properties work in traits.

"""
from __future__ import division

import sympy as sp

from traits.api import (cached_property, Dict, Float, HasPrivateTraits, Instance,
                        Property, Str)

import inputs

class Model(HasPrivateTraits):
    """Abstract class representing the model."""

    _numeric_F = Property(depends_on='_symbolic_F')

    _numeric_Fl = Property(depends_on='_symbolic_Fl')

    _numeric_Flr = Property(depends_on='_symbolic_Flr')

    _numeric_Fr = Property(depends_on='_symbolic_Fr')

    _numeric_Fx = Property(depends_on='_symbolic_Fx')

    _numeric_Fxr = Property(depends_on='_symbolic_Fxr')

    _numeric_Fxy = Property(depends_on='_symbolic_Fxy')

    _numeric_Fy = Property(depends_on='_symbolic_Fy')

    _numeric_Fyl = Property(depends_on='_symbolic_Fyl')

    _numeric_jacobian = Property(depends_on='_symbolic_jacobian')

    _numeric_ode_system = Property(depends_on='_symbolic_ode_system')

    _symbolic_F = Property(depends_on=['output, params'])

    _symbolic_Fl = Property(depends_on='_symbolic_F')

    _symbolic_Flr = Property(depends_on='_symbolic_F')

    _symbolic_Fr = Property(depends_on='_symbolic_F')

    _symbolic_Fx = Property(depends_on='_symbolic_F')

    _symbolic_Fxr = Property(depends_on='_symbolic_F')

    _symbolic_Fxy = Property(depends_on='_symbolic_F')

    _symbolic_Fy = Property(depends_on='_symbolic_F')

    _symbolic_Fyl = Property(depends_on='_symbolic_F')

    _symbolic_H = Property

    _symbolic_jacobian = Property(depends_on=['_symbolic_ode_system'])

    _symbolic_mu_prime= Property

    _symbolic_ode_system = Property(depends_on='matching')

    _symbolic_theta_prime = Property

    firms = Instance(inputs.Input)

    # need to validate this trait!
    matching = Str('PAM')

    # not sure this class capture all possible use cases!
    output = Instance(sp.Mul)

    params = Dict(Str, Float)

    workers = Instance(inputs.Input)

    @cached_property
    def _get__numeric_F(self):
        """Numeric function defining output."""
        args = sp.var('x, y, l, r')
        return sp.lambdify(args, self._symbolic_F, modules='numpy')

    @cached_property
    def _get__numeric_Fx(self):
        """Numeric function defining marginal product of x."""
        args = sp.var('x, y, l, r')
        return sp.lambdify(args, self._symbolic_Fx, modules='numpy')

    @cached_property
    def _get__numeric_Fxr(self):
        """Numeric function for cross-partial Fxr."""
        args = sp.var('x, y, l, r')
        return sp.lambdify(args, self._symbolic_Fxr, modules='numpy')

    @cached_property
    def _get__numeric_Fxy(self):
        """Numeric function for cross-partial Fxy."""
        args = sp.var('x, y, l, r')
        return sp.lambdify(args, self._symbolic_Fxy, modules='numpy')

    @cached_property
    def _get__numeric_Fy(self):
        """Numeric function defining marginal product of y."""
        args = sp.var('x, y, l, r')
        return sp.lambdify(args, self._symbolic_Fy, modules='numpy')

    @cached_property
    def _get__numeric_Fyl(self):
        """Numeric function for cross-partial Fyl."""
        args = sp.var('x, y, l, r')
        return sp.lambdify(args, self._symbolic_Fyl, modules='numpy')

    @cached_property
    def _get__numeric_Fl(self):
        """Numeric function defining marginal product of l."""
        args = sp.var('x, y, l, r')
        return sp.lambdify(args, self._symbolic_Fl, modules='numpy')

    @cached_property
    def _get__numeric_Fr(self):
        """Numeric function defining marginal product of r."""
        args = sp.var('x, y, l, r')
        return sp.lambdify(args, self._symbolic_Fr, modules='numpy')

    @cached_property
    def _get__numeric_jacobian(self):
        """Numeric function defining the Jacobian."""
        args = sp.var('x, mu, theta')
        return sp.lambdify(args, self._symbolic_jacobian, modules='numpy')

    @cached_property
    def _get__numeric_ode_system(self):
        """Numeric function defining the system of ODEs."""
        args = sp.var('x, mu, theta')
        return sp.lambdify(args, self._symbolic_ode_system, modules='numpy')

    @cached_property
    def _get__symbolic_F(self):
        """Symbolic expression for output."""
        return self.output.subs(self.params)

    @cached_property
    def _get__symbolic_Fx(self):
        """Symbolic expression for marginal product of x."""
        sp.var('x')
        return sp.diff(self._symbolic_F, x)

    @cached_property
    def _get__symbolic_Fxr(self):
        """Symbolic expression for cross-partial Fxr."""
        sp.var('x, r')
        return sp.diff(self._symbolic_F, x, 1, r, 1)

    @cached_property
    def _get__symbolic_Fxy(self):
        """Symbolic expression for cross-partial Fxy."""
        sp.var('x, y')
        return sp.diff(self._symbolic_F, x, 1, y, 1)

    @cached_property
    def _get__symbolic_Fy(self):
        """Symbolic expression for marginal product of y."""
        sp.var('y')
        return sp.diff(self._symbolic_F, y)

    @cached_property
    def _get__symbolic_Fyl(self):
        """Symbolic expression for cross-partial Fyl."""
        sp.var('l, y')
        return sp.diff(self._symbolic_F, y, 1, l, 1)

    @cached_property
    def _get__symbolic_Fl(self):
        """Symbolic expression for marginal product of l."""
        sp.var('l')
        return sp.diff(self._symbolic_F, l)

    @cached_property
    def _get__symbolic_Flr(self):
        """Symbolic expression for cross-partial Flr."""
        sp.var('l, r')
        return sp.diff(self._symbolic_F, l, 1, r, 1)

    @cached_property
    def _get__symbolic_Fr(self):
        """Symbolic expression for marginal product of r."""
        sp.var('r')
        return sp.diff(self._symbolic_F, r)

    def _get__symbolic_H(self):
        """Density ratio of worker skill to firm productivity."""
        sp.var('mu')

        # firm productivity is match specific!
        tmp_subs = {'x':mu}
        H = (self.workers.symbolic_normalized_pdf / 
             self.firms.symbolic_normalized_pdf.subs(tmp_subs))
        
        return H 

    @cached_property
    def _get__symbolic_jacobian(self):
        """Symbolic expression for the Jacobian."""
        args = sp.var('mu, theta')
        jacobian = self._symbolic_ode_system.jacobian(args)
        return jacobian

    def _get__symbolic_mu_prime(self):
        """Symbolic expressions for the ODE governing matching assignments."""
        sp.var('theta')

        if self.matching.lower() == 'nam':
            RHS = -self._symbolic_H / theta
        elif self.matching.lower() == 'pam':
            RHS = self._symbolic_H / theta
        else:
            raise ValueError

        return RHS

    def _get__symbolic_ode_system(self):
        """Symbolic expression for the system of ODEs."""
        args = sp.var('mu, theta')
        system = sp.Matrix([[self._symbolic_theta_prime],
                            [self._symbolic_mu_prime]])
        return system

    def _get__symbolic_theta_prime(self):
        """Symbolic expressions for the ODE governing factor intensity."""
        sp.var('mu, theta')

        # substitute for matching function mu and factor intensity theta
        tmp_subs = {'y':mu, 'l':theta, 'r':1.0}
        Fyl = self._symbolic_Fyl.subs(tmp_subs)
        Fxr = self._symbolic_Fxr.subs(tmp_subs)
        Flr = self._symbolic_Flr.subs(tmp_subs)

        # mu has already been substituted into density ratio
        H = self._symbolic_H
        
        if self.matching.lower() == 'nam':
            RHS = -(H * Fyl + Fxr) / Flr 
        elif self.matching.lower() == 'pam':
            RHS = (H * Fyl - Fxr) / Flr
        else:
            raise ValueError

        return RHS

        
        
if __name__ == '__main__':
    import numpy as np
    from scipy import stats

    from inputs import Input

    params = {'nu':0.89, 'kappa':1.0, 'gamma':0.54, 'rho':0.24, 'A':1.0}

    sp.var('A, kappa, nu, x, rho, y, l, gamma, r')
    F = r * A * kappa * (nu * x**rho + (1 - nu) * (y * (l / r))**rho)**(gamma / rho)

    # suppose worker skill is log normal
    sp.var('x, mu, sigma')
    skill_cdf = 0.5 + 0.5 * sp.erf((sp.log(x) - mu) / sp.sqrt(2 * sigma**2))
    worker_params = {'mu':0.0, 'sigma':1}
    x_lower, x_upper, x_measure = 1e-3, 5e1, 0.025
    
    workers = Input(distribution=skill_cdf,
                    lower_bound=x_lower,
                    params=worker_params,
                    upper_bound=x_upper,
                   )

    # suppose firm productivity is log normal
    sp.var('x, mu, sigma')
    productivity_cdf = 0.5 + 0.5 * sp.erf((sp.log(x) - mu) / sp.sqrt(2 * sigma**2))
    firm_params = {'mu':0.0, 'sigma':1}
    y_lower, y_upper, y_measure = 1e-3, 5e1, 0.025
    
    firms = Input(distribution=productivity_cdf,
                  lower_bound=x_lower,
                  params=firm_params,
                  upper_bound=x_upper,
                  )

    # create an instance of the Model class
    model = Model(firms=firms,
                  matching='PAM',
                  output=F,
                  params=params,
                  workers=workers
                  )

    print(model._numeric_jacobian(x_upper, 1e2, y_upper))
    print(model._numeric_ode_system(x_upper, 1e2, y_upper))
