"""
Test suite for the model.py module.

@author : David R. Pugh
@date : 2014-11-11

"""
import nose

import sympy as sym
from scipy import stats

import inputs
import model

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


def test__validate_assortativity():
    """Testing validation of assortativity attribute."""

    # assortativity must be either 'positive' or 'negative'
    invalid_assortativity = 'invalid_assortativity'

    with nose.tools.assert_raises(AttributeError):
        model.Model(invalid_assortativity, workers, firms, valid_F,
                    valid_params)

    # assortativity must be a string
    invalid_assortativity = 0.0

    with nose.tools.assert_raises(AttributeError):
        model.Model(invalid_assortativity, workers, firms, valid_F,
                    valid_params)


def test__validate_production_function():
    """Testing validation of production function attribute."""

    # production function must have type sym.Basic
    def invalid_F(x, y, l, r, A, kappa, nu, rho, gamma):
        """Valid F must return a SymPy expression."""
        return r * A * kappa * (nu * x**rho + (1 - nu) * (y * (l / r))**rho)**(gamma / rho)

    with nose.tools.assert_raises(AttributeError):
        model.Model('positive', workers, firms, production=invalid_F,
                    params=valid_params)

    # production function must share vars with workers and firms
    m, n = sym.var('m, n')
    invalid_F = r * A * kappa * (nu * m**rho + (1 - nu) * (n * (l / r))**rho)**(gamma / rho)

    with nose.tools.assert_raises(AttributeError):
        model.Model('negative', workers, firms, production=invalid_F,
                    params=valid_params)

    # production function must depend on r and l
    m, n = sym.var('m, n')
    invalid_F = r * A * kappa * (nu * x**rho + (1 - nu) * (y * (m / n))**rho)**(gamma / rho)

    with nose.tools.assert_raises(AttributeError):
        model.Model('positive', workers, firms, production=invalid_F,
                    params=valid_params)


def test__validate_params():
    """Testing validation of parameters attribute."""

    # valid parameters must be a dict
    invalid_params = [1.0, 2.0, 3.0, 4.0, 5.0]

    with nose.tools.assert_raises(AttributeError):
        model.Model('positive', workers, firms, production=valid_F,
                    params=invalid_params)


def test__validate_input():
    """Testing validation of inputs."""

    # valid inputs must be an instance of the Input class
    invalid_workers = stats.lognorm(s=1.0)
    invalid_firms = stats.lognorm(s=1.0)

    with nose.tools.assert_raises(AttributeError):
        model.Model('positive', invalid_workers, invalid_firms,
                    production=valid_F, params=valid_params)


def test__validate_model():
    """Testing validation of the model attribute."""
    valid_model = model.Model('positive', workers, firms, production=valid_F,
                              params=valid_params)

    # model attribute must be instance of model.Model
    invalid_model = 'not a model instance'
    with nose.tools.assert_raises(AttributeError):
        model.DifferentiableMatching(model=invalid_model)

    # confirm valid model attribute
    matching = model.DifferentiableMatching(model=valid_model)
    nose.tools.assert_equals(matching.model.params, valid_params)


def test_marginal_product_worker_skill():
    """Testing the Fx attribute."""
    # should be instance of sympy.Basic
    mod = model.Model('negative', workers, firms, valid_F, valid_params)
    nose.tools.assert_is_instance(mod.Fx, sym.Basic)


def test_skill_complementarity():
    """Testing the Fxy attribute."""
    # should be instance of sympy.Basic
    mod = model.Model('positive', workers, firms, valid_F, valid_params)
    nose.tools.assert_is_instance(mod.Fxy, sym.Basic)


def test_intensive_output():
    """Testing symbolic expression for intensive output."""
    mod = model.Model('positive', workers, firms, valid_F, valid_params)

    # y, l, and r should not appear in intensive output
    for var in [y, l, r]:
        nose.tools.assert_false(var in mod.matching.f.atoms())

    # mu and theta should appear in both mu_prime or theta_prime
    nose.tools.assert_true({mu, theta} < mod.matching.f.atoms())


def test_wages():
    """Testing symbolic expression for workers wages."""
    mod = model.Model('positive', workers, firms, valid_F, valid_params)

    # y, l, and r should not appear in workers wages
    for var in [y, l, r]:
        nose.tools.assert_false(var in mod.matching.w.atoms())

    # mu and theta should appear in both mu_prime or theta_prime
    nose.tools.assert_true({mu, theta} < mod.matching.w.atoms())


def test_mu_prime():
    """Testing symbolic expression for matching differential equation."""
    mod1 = model.Model('positive', workers, firms, valid_F, valid_params)
    mod2 = model.Model('negative', workers, firms, valid_F, valid_params)

    # mu_prime not implemented for base DifferentiableMatching class
    matching = model.DifferentiableMatching(mod1)
    with nose.tools.assert_raises(NotImplementedError):
        matching.mu_prime

    # y, l, and r should not appear in either mu_prime or theta_prime
    for var in [y, l, r]:
        nose.tools.assert_false(var in mod1.matching.mu_prime.atoms())
        nose.tools.assert_false(var in mod2.matching.mu_prime.atoms())

    # mu and theta should appear in both mu_prime or theta_prime
    nose.tools.assert_true({mu, theta} < mod1.matching.mu_prime.atoms())
    nose.tools.assert_true({mu, theta} < mod2.matching.mu_prime.atoms())


def test_theta_prime():
    """Testing symbolic expression for firm size differential equation."""
    mod1 = model.Model('negative', workers, firms, valid_F, valid_params)
    mod2 = model.Model('positive', workers, firms, valid_F, valid_params)

    # theta_prime not implemented in base DifferentiableMatching class
    matching = model.DifferentiableMatching(mod1)
    with nose.tools.assert_raises(NotImplementedError):
        matching.theta_prime

    # y, l, and r should not appear in either mu_prime or theta_prime
    for var in [y, l, r]:
        nose.tools.assert_false(var in mod1.matching.theta_prime.atoms())
        nose.tools.assert_false(var in mod2.matching.theta_prime.atoms())

    # mu and theta should appear in both mu_prime or theta_prime
    nose.tools.assert_true({mu, theta} < mod1.matching.theta_prime.atoms())
    nose.tools.assert_true({mu, theta} < mod2.matching.theta_prime.atoms())