import numpy as np
from scipy import optimize


class OrthogonalCollocation(object):
    """Base class for OrthogonalCollocation."""

    @staticmethod
    def _coefs_array_to_list(coefs_array, degrees):
        """Spliny array of coefficients into list of coefficient arrays."""
        coefs_list = []
        for degree in degrees:
            coefs_list.append(coefs_array[:degree])
            coefs_array = coefs_array[degree:]
        return coefs_list

    @staticmethod
    def _coefs_list_to_array(coefs_list):
        """Combine list of coefficient arrays into array of coefficients."""
        return np.hstack(coefs_list)

    @staticmethod
    def _orthogonal_polynomial_factory(kind, coef, domain):
        """Returns an orthogonal polynomial given some coefficients."""
        if kind == "Chebyshev":
            polynomial = np.polynomial.Chebyshev(coef, domain)
        elif kind == "Hermite":
            polynomial = np.polynomial.Hermite(coef, domain)
        elif kind == "Laguerre":
            polynomial = np.polynomial.Laguerre(coef, domain)
        elif kind == "Legendre":
            polynomial = np.polynomial.Legendre(coef, domain)
        else:
            valid_kinds = ['Chebyshev', 'Hermite', 'Legendre', 'Laguerre']
            error_mesg = "Parameter 'kind' must be one of {}, {}, {}, or {}"
            raise ValueError(error_mesg.format(*valid_kinds))

        return polynomial


class OrthogonalCollocationSolver(OrthogonalCollocation):

    def solve(self, kind, coefs, domain, method="hybr", **kwargs):
        initial_guess = np.hstack(coefs)
        result = optimize.root(self._evaluate_collocation_residuals,
                               x0=initial_guess,
                               args=(kind, domain),
                               method=method,
                               **kwargs)
        return result

    def _evaluate_collocation_residuals(self, coefs_array, kind, domain):
        polynomials = self._construct_polynomials(kind, coefs_array, domain)
        derivatives = self._construct_derivatives(polynomials)
        residuals = self._construct_residuals(polynomials, derivatives)

    def _construct_polynomials(self, kind, coefs, domain):
        return [self._orthogonal_polynomial_factory(kind, coefs, domain)]
