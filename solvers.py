import sympy as sym

import models


# should really use letters for variables so as not to confound with params!
mu, theta = sym.var('mu, theta')


class ShootingSolver(object):
    """Solves a model using forward shooting."""

    def __init__(self, model):
        self.model = model

    @property
    def _symbolic_equations(self):
        return [self.model.matching.mu_prime, self.model.matching.theta_prime]

    @property
    def _symbolic_jacobian(self):
        return self._symbolic_system.jacobian([mu, theta])

    @property
    def _symbolic_system(self):
        return sym.Matrix(self._symbolic_equations)


if __name__ == '__main__':
    import inputs

    # define endogenous variables
    mu, theta = sym.var('mu, theta')

    # define some workers skill
    x, mu1, sigma1 = sym.var('x, mu1, sigma1')
    skill_cdf = 0.5 + 0.5 * sym.erf((sym.log(x) - mu1) / sym.sqrt(2 * sigma1**2))
    skill_params = {'mu1': 0.0, 'sigma1': 1.0}
    skill_bounds = [1e-3, 5e1]

    workers = inputs.Input(var=x,
                           cdf=skill_cdf,
                           params=skill_params,
                           bounds=skill_bounds,
                           )

    # define some firms
    y, mu2, sigma2 = sym.var('y, mu2, sigma2')
    productivity_cdf = 0.5 + 0.5 * sym.erf((sym.log(y) - mu2) / sym.sqrt(2 * sigma2**2))
    productivity_params = {'mu2': 0.0, 'sigma2': 1.0}
    productivity_bounds = [1e-3, 5e1]

    firms = inputs.Input(var=y,
                         cdf=productivity_cdf,
                         params=productivity_params,
                         bounds=productivity_bounds,
                         )

    # define some valid model params
    valid_params = {'nu': 0.89, 'kappa': 1.0, 'gamma': 0.54, 'rho': 0.24, 'A': 1.0}

    # define a valid production function
    A, kappa, nu, rho, l, gamma, r = sym.var('A, kappa, nu, rho, l, gamma, r')
    valid_F = r * A * kappa * (nu * x**rho + (1 - nu) * (y * (l / r))**rho)**(gamma / rho)

    valid_model = models.Model('positive', workers, firms, production=valid_F,
                               params=valid_params)

    solver = ShootingSolver(model=valid_model)
