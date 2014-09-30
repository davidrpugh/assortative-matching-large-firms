"""
Test functions for the inputs.py module.

@author : David R. Pugh
@date : 2014-09-30

"""
import nose

import numpy as np
from scipy import stats
import sympy as sym

import inputs

# define a valid cdf expression
valid_var, mu, sigma = sym.var('x, mu, sigma')
valid_cdf = 0.5 + 0.5 * sym.erf((sym.log(valid_var) - mu) / sym.sqrt(2 * sigma**2))
valid_params = {'mu': 0.0, 'sigma': 1.0}
valid_bounds = [1e-3, 5e1]


def test_validate_cdf():
    """Testing validation of cdf attribute."""

    def invalid_cdf(x, mu, sigma):
        """Valid cdf must return a SymPy expression."""
        return stats.lognorm.cdf(x, sigma, scale=np.exp(mu))

    with nose.tools.assert_raises(AttributeError):
        inputs.Input(var=valid_var, cdf=invalid_cdf, bounds=valid_bounds,
                     params=valid_params)


def test_validate_lower():
    """Testing validation of lower attribute."""

    # lower should be a float
    invalid_lower = 1

    with nose.tools.assert_raises(AttributeError):
        workers = inputs.Input(var=valid_var, cdf=valid_cdf,
                               bounds=valid_bounds, params=valid_params)
        workers.lower = invalid_lower


def test_validate_upper():
    """Testing validation of upper attribute."""

    # upper should be a float
    invalid_upper = 14

    with nose.tools.assert_raises(AttributeError):
        workers = inputs.Input(var=valid_var, cdf=valid_cdf,
                               bounds=valid_bounds, params=valid_params)
        workers.upper = invalid_upper


def test_validate_params():
    """Testing validation of bounds parameter."""

    # valid parameters must be a dict
    invalid_params = (1.0, 2.0)

    with nose.tools.assert_raises(AttributeError):
        inputs.Input(var=valid_var, cdf=valid_cdf, bounds=valid_bounds,
                     params=invalid_params)


def test_validate_var():
    """Testing validation of var attribute."""

    # valid var must be a sym.Symbol
    invalid_var = 'x'

    with nose.tools.assert_raises(AttributeError):
        inputs.Input(var=invalid_var, cdf=valid_cdf, bounds=valid_bounds,
                     params=valid_params)


def test_evaluate_cdf():
    """Testing the evaluation of the cdf."""

    # suppose that workers are uniform on [a, b] = [0, 1]
    a, b = sym.var('a, b')
    uniform_cdf = (valid_var - a) / (b - a)
    params = {'a': 0.0, 'b': 1.0}
    workers = inputs.Input(var=valid_var, cdf=uniform_cdf, bounds=[0.0, 1.0],
                           params=params)

    # evaluate with scalar input
    actual_cdf = workers.evaluate_cdf(0.5)
    expected_cdf = 0.5
    nose.tools.assert_almost_equals(actual_cdf, expected_cdf)

    # evaluate with array input
    actual_cdf = workers.evaluate_cdf(np.array([0.0, 0.5, 1.0]))
    expected_cdf = actual_cdf
    np.testing.assert_almost_equal(actual_cdf, expected_cdf)


def test_evaluate_pdf():
    """Testing the evaluation of the pdf."""

    # suppose that workers are uniform on [a, b] = [0, 1]
    a, b = sym.var('a, b')
    uniform_cdf = (valid_var - a) / (b - a)
    params = {'a': 0.0, 'b': 1.0}
    workers = inputs.Input(var=valid_var, cdf=uniform_cdf, bounds=[0.25, 0.75],
                           params=params)

    # evaluate with scalar input and norm=False
    actual_pdf = workers.evaluate_pdf(0.5, norm=False)
    expected_pdf = 1.0
    nose.tools.assert_almost_equals(actual_pdf, expected_pdf)

    # evaluate with array input and norm=False
    actual_pdf = workers.evaluate_pdf(np.array([0.25, 0.5, 0.75]), norm=False)
    expected_pdf = np.ones(3)
    np.testing.assert_almost_equal(actual_pdf, expected_pdf)

    # evaluate with scalar input and norm=True
    actual_pdf = workers.evaluate_pdf(0.5, norm=True)
    expected_pdf = 2.0
    nose.tools.assert_almost_equals(actual_pdf, expected_pdf)

    # evaluate with array input and norm=False
    actual_pdf = workers.evaluate_pdf(np.array([0.25, 0.5, 0.75]), norm=True)
    expected_pdf = np.repeat(2.0, 2)
    np.testing.assert_almost_equal(actual_pdf, expected_pdf)