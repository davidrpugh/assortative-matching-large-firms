import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import sympy as sym

import inputs
import models
import shooting

def polynomial_factory(coefficients, kind, domain):
        """
        Factory function for generating various orthogonal polynomials.

        Parameters
        ----------
        coefficients : numpy.ndarray (shape=(N,))
            Array of polynomial coefficients.
        kind : string
            Class of orthogonal polynomials to use as basic functions. Must be
            one of "Chebyshev", "Hermite", "Laguerre", or "Legendre."
        domain: list
        	Must be [self.model.workers.lower, self.model.workers.upper]

        Returns
        -------
        polynomial : numpy.polynomial.Polynomial
            Approximating polynomial.

        """
        if kind == "Chebyshev":
            polynomial = np.polynomial.Chebyshev(coefficients, domain)
        elif kind == "Hermite":
            polynomial = np.polynomial.Hermite(coefficients, domain)
        elif kind == "Laguerre":
            polynomial = np.polynomial.Laguerre(coefficients, domain)
        elif kind == "Legendre":
            polynomial = np.polynomial.Legendre(coefficients, domain)
        else:
            mesg = ("Somehow you managed to specify an invalid 'kind' of " +
                    "orthogonal polynomials!")
            raise ValueError(mesg)
        return polynomial


def Solve_Model(Fnc, F_params, workers, firms, ass, intg, ini):
	"""
	Function that solves the sorting model and returns wage and size distributions

	Assumes monotonicity,log normal distributions for size and wages.
	To fit the distributions, it uses firm skill (and not worked skill) as main variable.
	It also only uses the shooting solver, checking for completition.

	Parameters
    ----------
	Fnc: Production Function (sym)
	F_params: Parameters of Fnc (dic)
	workers: an Input class instance for workers
	firms: an Input class instance for firms
	ass: assortativity ('positive' or 'negative') (string)
	intg: a string with the integrator of preference (string)
	ini: initial guess (float)

	Returns
    -------
	A tutple consisting on (string label,(distribution parameters), fitted values)

	"""
	flag_solver = False
	modelA = models.Model(assortativity=ass,
    	                 workers=workers,
        	             firms=firms,
            	         production=Fnc,
                	     params=F_params)

	solver = shooting.ShootingSolver(model=modelA)
	''' 1.Solve the Model '''
	solver.solve(ini, tol=1e-6, number_knots=1000, integrator=intg,
        	     atol=1e-12, rtol=1e-9)
	
	''' 2.Check it is truly solved '''
	if not (solver._converged_firms(1e-6) and solver._converged_firms(1e-6)):
		flag_solver = True
	err_mesg = ("Fail to solve!")
   	assert (solver._converged_firms(1e-6) and solver._converged_firms(1e-6)), err_mesg

   	''' 3.Store vectors of results '''
   	x = solver.solution.index.values
	mus = solver.solution['$\\mu(x)$'].values
	thetas = solver.solution['$\\theta(x)$'].values
	ws = solver.solution['$w(x)$'].values

	''' 4.Interpolate the other way round (y(theta), y(wages)) '''
	basePol1 = polynomial_factory(20, "Chebyshev", [firms.lower, firms.upper])
	basePol2 = polynomial_factory(20, "Chebyshev", [firms.lower, firms.upper])
	y_theta = basePol1.fit(thetas, mus, 20)
	y_wage = basePol2.fit(ws, mus, 20)

	''' 5. Transform into distibutions '''
	theta_range = np.arange(min(thetas), max(thetas), 0.01)
	w_range = np.arange(min(ws), max(ws), 0.01)
	theta_dis = firms.evaluate_pdf(y_theta(theta_range))
	wage_dis = firms.evaluate_pdf(y_wage(w_range))

	''' 6. Fit distributions '''
	shapel, locationl, scalel = stats.lognorm.fit(theta_dis)
	shapew, locationw, scalew = stats.lognorm.fit(wage_dis)

	return ('theta',(shapel, locationl, scalel), theta_dis), ('w',(shapew, locationw, scalew),wage_dis)

def Calculate_MSE_pdf(data, data_pams, estimated_pams, distribution):
	'''
	Given some estimated shape distribution parameters, gives the mean square error related to
	the data's fitted distribution.

	Must have the module stats imported from scipy!!

	Parameters
    ----------
    data:			data points to calculate the mse (tup of np.ndarrays)
	data_pams: 		parameters of data distributions (tup of floats)
	estimated_pams: parameters of model distributions (tup of floats)
	distribution: 	type of distribution of the n distributions to estimated (str)
					(Supported by now: 'lognormal' only)

	Returns
    -------
	mse = Mean square errors of fit (tup of floats, one for each n)
	'''
	mse = ()
	if distribution =='lognormal':
		for i in range(len(data)):
			shape, loc, scale = data_pams[i]
			eshape, eloc, escale = estimated_pams[i]
			err_sq = (stats.lognorm.pdf(data[i],s=shape, loc=loc, scale=scale) - stats.lognorm.pdf(data[i],s=eshape, loc=eloc, scale=escale))**2
			mse += (sum(err_sq)/float(len(err_sq)),)
	else:
		raise NotImplementedError

	return mse

def Calculate_MSE_cdf(data, data_pams, estimated_pams, distribution):
	'''
	Given some estimated shape distribution parameters, gives the mean square error related to
	the data's fitted distribution.

	Must have the module stats imported from scipy!!

	Parameters
    ----------
    data:			data points to calculate the mse (tup of np.ndarrays)
	data_pams: 		parameters of data distributions (tup of floats)
	estimated_pams: parameters of model distributions (tup of floats)
	distribution: 	type of distribution of the n distributions to estimated (str)
					(Supported by now: 'lognormal' only)

	Returns
    -------
	mse = Mean square errors of fit (tup of floats, one for each n)
	'''
	mse = ()
	if distribution =='lognormal':
		for i in range(len(data)):
			shape, loc, scale = data_pams[i]
			eshape, eloc, escale = estimated_pams[i]
			err_sq = (stats.lognorm.cdf(data[i],s=shape, loc=loc, scale=scale) - stats.lognorm.cdf(data[i],s=eshape, loc=eloc, scale=escale))**2
			mse += (sum(err_sq)/float(len(err_sq)),)
	else:
		raise NotImplementedError

	return mse
