import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np
import sympy as sym

import inputs
import models
import shooting



def Solve_Model(Fnc, F_params, workers, firms, ass, N_knots, intg, ini):
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
	N_knots: number of knots (int)
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
	solver.solve(ini, tol=1e-6, number_knots=N_knots, integrator=intg,
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
	sateht = thetas[::-1]
	sw = ws[::-1]
	um = mus[::-1]

	if thetas[0]>thetas[-1]:
		y_theta = InterpolatedUnivariateSpline(sateht, um)
	else:
		y_theta = InterpolatedUnivariateSpline(thetas, mus)
	
	if ws[0]>ws[-1]:
		y_wage = InterpolatedUnivariateSpline(sw, um)
	else:
		y_wage = InterpolatedUnivariateSpline(ws, mus)

	''' 5. Transform into distibutions '''
	if thetas[0]>thetas[-1]:
		t_step = (max(sateht[:-1])-min(sateht))/N_knots
		theta_range = np.arange(min(sateht), max(sateht[:-1]), t_step)
	else:
		t_step = (max(thetas[1:])-min(thetas))/N_knots
		theta_range = np.arange(min(thetas), max(thetas[:-1]), t_step)
	
	if ws[0]>ws[-1]:
		w_step = (max(sw[:-1])-min(sw))/N_knots
		w_range = np.arange(min(sw), max(sw[:-1]), w_step)
	else:
		w_step = (max(ws[:-1])-min(ws))/N_knots
		w_range = np.arange(min(ws), max(ws[:-1]), w_step)
	
	theta_dis1 = firms.evaluate_pdf(y_theta(theta_range))
	wage_dis1 = firms.evaluate_pdf(y_wage(w_range))

	''' 5.b Clean nans'''
	theta_dis = np.nan_to_num(theta_dis1)
	wage_dis = np.nan_to_num(wage_dis1)

	''' 6. Fit distributions '''
	shapel, locationl, scalel = stats.lognorm.fit(theta_dis)
	shapew, locationw, scalew = stats.lognorm.fit(wage_dis)

	return (shapel, locationl, scalel), (shapew, locationw, scalew)

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
			err_sq = ((stats.lognorm.pdf(data[i],s=shape, loc=loc, scale=scale) - stats.lognorm.pdf(data[i],s=eshape, loc=eloc, scale=escale))*100)**2
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
			err_sq = ((stats.lognorm.cdf(data[i],s=shape, loc=loc, scale=scale) - stats.lognorm.cdf(data[i],s=eshape, loc=eloc, scale=escale))*100)**2
			mse += (sum(err_sq)/float(len(err_sq)),)
	else:
		raise NotImplementedError

	return mse
