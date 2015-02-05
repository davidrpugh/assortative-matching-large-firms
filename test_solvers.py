"""
Testing suite for the solver.py module.

@author : David R. Pugh
@date : 2014-11-12

"""
import unittest

import numpy as np
import sympy as sym

import inputs
import models
import shooting


class MultiplicativeSeparabilityCase(unittest.TestCase):

    def setUp(self):
        """Set up code for test fixtures."""
        # define some workers skill
        x, a, b = sym.var('x, a, b')
        skill_cdf = (x - a) / (b - a)
        skill_params = {'a': 1.0, 'b': 2.0}
        skill_bounds = [skill_params['a'], skill_params['b']]

        workers = inputs.Input(var=x,
                               cdf=skill_cdf,
                               params=skill_params,
                               bounds=skill_bounds,
                               )

        # define some firms
        y = sym.var('y')
        productivity_cdf = (y - a) / (b - a)
        productivity_params = skill_params
        productivity_bounds = skill_bounds

        firms = inputs.Input(var=y,
                             cdf=productivity_cdf,
                             params=productivity_params,
                             bounds=productivity_bounds,
                             )

        # define symbolic expression for CES between x and y
        x, y, omega_A, sigma_A = sym.var('x, y, omega_A, sigma_A')
        A = ((omega_A * x**((sigma_A - 1) / sigma_A) +
             (1 - omega_A) * y**((sigma_A - 1) / sigma_A))**(sigma_A / (sigma_A - 1)))

        # define symbolic expression for Cobb-Douglas between l and r
        l, r, omega_B, sigma_B = sym.var('l, r, omega_A, sigma_A')
        B = l**omega_B * r**(1 - omega_B)

        # multiplicative separability!
        F_params = {'omega_A': 0.5, 'omega_B': 0.5, 'sigma_A': 0.5, 'sigma_B': 1.0}
        F = A * B

        self.model = models.Model(assortativity='positive',
                                  workers=workers,
                                  firms=firms,
                                  production=F,
                                  params=F_params)

        self.solver = shooting.ShootingSolver(model=self.model)

    def test_solve(self):
        """Test trivial example for solver."""
        # approach solution from above
        guess_firm_size_upper = 2.5
        self.solver.solve(guess_firm_size_upper, tol=1e-9, number_knots=10,
                          atol=1e-15, rtol=1e-12)

        # conduct the test
        T = self.solver.solution.shape[0]
        expected_theta = np.ones(T)
        actual_theta = self.solver.solution['firm size']
        np.testing.assert_almost_equal(expected_theta, actual_theta)

        # approach solution from below
        guess_firm_size_upper = 1.5
        self.solver.solve(guess_firm_size_upper, tol=1e-9, number_knots=10,
                          integrator='lsoda', with_jacobian=True, atol=1e-15,
                          rtol=1e-12)

        # conduct the test
        T = self.solver.solution.shape[0]
        expected_theta = np.ones(T)
        actual_theta = self.solver.solution['firm size']
        np.testing.assert_almost_equal(expected_theta, actual_theta)
