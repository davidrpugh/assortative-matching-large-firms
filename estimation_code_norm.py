import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import PchipInterpolator
import numpy as np
import sympy as sym
import csv

import inputs
import models
import shooting



def Solve_Model(Fnc, F_params, workers, firms, ass, N_knots, intg, ini, tolerance=1e-6):
	"""
	Function that solves the sorting model and returns functions mu(x), theta(x), w(x).

	Assumes monotonicity,log normal distributions for size and wages.
	To fit the data, it uses worker skill as main variable.
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
	A tutple consisting on (mu(x)), theta(x). w(x)), final result from 

	"""
	flag_solver = False
	modelA = models.Model(assortativity=ass,
    	                 workers=workers,
        	             firms=firms,
            	         production=Fnc,
                	     params=F_params)

	solver = shooting.ShootingSolver(model=modelA)
	''' 1.Solve the Model '''
	solver.solve(ini, tol=tolerance, number_knots=N_knots, integrator=intg,
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

	''' 4.Interpolate mu(x), theta(x), w(x) '''
	mu_x = PchipInterpolator(x, mus)
	theta_x = PchipInterpolator(x, thetas)
	w_x = PchipInterpolator(x, ws)

	
	return (mu_x, theta_x, w_x), thetas[-1]

def import_data(file_name, ID=True):
    '''
    This function imports the data from a csv file, returns ndarrays with it

    Input
    -----

    file_name : (str) path and name of the csv file.
    ID (optional) : boolean, True if it contais ID in the first collunm

    Output
    ------
    Following np.ndarrays: firmID (str), size (int), wage (float), profit (float), skill_w (float), 
    firm_age (float), industry_code (int), region (int)

    '''
    # Opening data
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
       	data = list(reader)

    # Passing data to lists, then to arrays (should change this to make it all in one) 
    firmID = []
    size = []
    wage = []
    profit = []
    skill_w = []
    c = 0
    if ID==False:
    	c += 1
    for row in data[1:]:
        size.append(float(row[1-c]))
        wage.append(float(row[2-c]))
        profit.append(float(row[3-c]))
        skill_w.append(float(row[4-c]))
    # Firm size in workers (int)
    size = np.asarray(size)
    # Daily average wage for each firm, in euros (float)
    wage = np.asarray(wage)
    # Declared average profits for each firm per year, in euros (float)
    profit = np.asarray(profit)
    # Average education level of workers per firm, from 0 to 6 (float)
    skill_w = np.asarray(skill_w)

    return skill_w, profit, size, wage


def Calculate_MSE(data, functions_from_model):
	'''
	Given some estimated shape distribution parameters, gives the mean square error related to
	the data's fitted distribution.

	Must have the module stats imported from scipy!!

	Parameters
    ----------
    data:			        (1x4) tuple of ndarrays. Data points to calculate the mse Order: x, y, theta, w.
	functions_from_model: 	(1x3) tuple of functions from the model solution. Order: mu(x), theta(x), w(x).
							Should be executable (that is, input a float and return a float)			

	Returns
    -------
	mse = Mean square errors of fit (tup of floats, one for each n)
	'''

	'''1. Unpack Data and functions'''
	xs = data[0]
	ys = data[1]
	thetas = data[2]
	ws = data[3]

	mu_x = functions_from_model[0]
	theta_x = functions_from_model[1]
	w_x = functions_from_model[2]

	'''2. Calculate Mean Square error'''

	mu_err = []
	theta_err = []
	w_err = []			# Should do this with arrays

	for i in range(len(xs)):
		mu_hat = mu_x(xs[i])
		theta_hat = theta_x(xs[i])
		w_hat = w_x(xs[i])

		mu_err.append((mu_hat-ys[i])**2)
		theta_err.append((theta_hat-thetas[i])**2)
		w_err.append((w_hat-ws[i])**2)

	return np.sum(mu_err+theta_err+w_err)


def ObjectiveFunction(params, data, x_pam, x_bounds, y_pam, y_bounds, guess):
	"""
	Calculates the sum of squared errors from the model with given parameters.

	Assumes lognormal distribution for both firms and workers.

	In: tuple of 3 parameters
		tuple of 4 ndarrays with data points for (x, y, theta, w)
		tuple with mean and variance of the worker distribution
		tuple with the bounds of the worker distribution
		tuple with mean and variance of the firm distribution
		tuple with the bounds of the firm distribution

	Out: Sum of squared residuals

	"""
	""" 1. Unpack workers and firms distributions """
	# define some default workers skill
	x, mu1, sigma1 = sym.var('x, mu1, sigma1')
	skill_cdf = 0.5 + 0.5 * sym.erf((x - mu1) / sym.sqrt(2 * sigma1**2))
	skill_params = {'mu1': x_pam[0], 'sigma1': x_pam[1]}
	skill_bounds = [x_bounds[0], x_bounds[1]]

	workers = inputs.Input(var=x,
                           cdf=skill_cdf,
                       		params=skill_params,
                       		bounds=skill_bounds,
                      		)

	# define some default firms
	y, mu2, sigma2 = sym.var('y, mu2, sigma2')
	productivity_cdf = 0.5 + 0.5 * sym.erf((y - mu2) / sym.sqrt(2 * sigma2**2))
	productivity_params = {'mu2': y_pam[0], 'sigma2': y_pam[1]}
	productivity_bounds = [y_bounds[0], y_bounds[1]]

	firms = inputs.Input(var=y,
    	                 cdf=productivity_cdf,
        	             params=productivity_params,
            	         bounds=productivity_bounds,
                	     )
	""" 2. Unpack params and define the production function (MS) """
	# CES between x and y
	omega_A, sigma_A = sym.var('omega_A, sigma_A')
	A = ((omega_A * x**((sigma_A - 1) / sigma_A) + 
    	 (1 - omega_A) * y**((sigma_A - 1) / sigma_A))**(sigma_A / (sigma_A - 1))) 

	# Cobb-Douglas between l and r
	l, r, omega_B = sym.var('l, r, omega_B')
	B = l**omega_B * r**(1 - omega_B)

	F = A * B

	F_params = {'omega_A':params[0], 'omega_B':params[1], 'sigma_A':params[2]}

	""" 3. Solve the model """
	sol = Solve_Model(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess)
	mu_hat, theta_hat, w_hat = sol[0]
	guess = sol[1]

	""" 4. Calculate and return """
	mse = Calculate_MSE(data, (mu_hat, theta_hat, w_hat) )

	return mse

def BulletProofObjectiveFunction(params, data, x_pam, x_bounds, y_pam, y_bounds, guess):
	"""
	Calculates the sum of squared errors from the model with given parameters.

	Assumes lognormal distribution for both firms and workers.

	In: tuple of 3 parameters
		tuple of 4 ndarrays with data points for (x, y, theta, w)
		tuple with mean and variance of the worker distribution
		tuple with the bounds of the worker distribution
		tuple with mean and variance of the firm distribution
		tuple with the bounds of the firm distribution

	Out: Sum of squared residuals

	"""
	""" 1. Unpack workers and firms distributions """
	# define some default workers skill
	x, mu1, sigma1 = sym.var('x, mu1, sigma1')
	skill_cdf = 0.5 + 0.5 * sym.erf((x - mu1) / sym.sqrt(2 * sigma1**2))
	skill_params = {'mu1': x_pam[0], 'sigma1': x_pam[1]}
	skill_bounds = [x_bounds[0], x_bounds[1]]

	workers = inputs.Input(var=x,
                           cdf=skill_cdf,
                       		params=skill_params,
                       		bounds=skill_bounds,
                      		)

	# define some default firms
	y, mu2, sigma2 = sym.var('y, mu2, sigma2')
	productivity_cdf = 0.5 + 0.5 * sym.erf((y - mu2) / sym.sqrt(2 * sigma2**2))
	productivity_params = {'mu2': y_pam[0], 'sigma2': y_pam[1]}
	productivity_bounds = [y_bounds[0], y_bounds[1]]

	firms = inputs.Input(var=y,
    	                 cdf=productivity_cdf,
        	             params=productivity_params,
            	         bounds=productivity_bounds,
                	     )
	""" 2. Unpack params and define the production function (MS) """
	# CES between x and y
	omega_A, sigma_A = sym.var('omega_A, sigma_A')
	A = ((omega_A * x**((sigma_A - 1) / sigma_A) + 
    	 (1 - omega_A) * y**((sigma_A - 1) / sigma_A))**(sigma_A / (sigma_A - 1))) 

	# Cobb-Douglas between l and r
	l, r, omega_B = sym.var('l, r, omega_B')
	B = l**omega_B * r**(1 - omega_B)

	F = A * B

	F_params = {'omega_A':params[0], 'omega_B':params[1], 'sigma_A':params[2]}

	""" 3. Solve the model """
	try:
		sol = Solve_Model(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess)
		mu_hat, theta_hat, w_hat = sol[0]
		guess = sol[1]

		""" 4. Calculate and return """
		mse = Calculate_MSE(data, (mu_hat, theta_hat, w_hat) )
	except AssertionError:
		mse = np.inf
	return mse

def StubbornObjectiveFunction(params, data, x_pam, x_bounds, y_pam, y_bounds, guess):
	"""
	Calculates the sum of squared errors from the model with given parameters.

	Assumes lognormal distribution for both firms and workers.

	In: tuple of 3 parameters
		tuple of 4 ndarrays with data points for (x, y, theta, w)
		tuple with mean and variance of the worker distribution
		tuple with the bounds of the worker distribution
		tuple with mean and variance of the firm distribution
		tuple with the bounds of the firm distribution

	Out: Sum of squared residuals

	"""
	""" 1. Unpack workers and firms distributions """
	# define some default workers skill
	x, mu1, sigma1 = sym.var('x, mu1, sigma1')
	skill_cdf = 0.5 + 0.5 * sym.erf((x - mu1) / sym.sqrt(2 * sigma1**2))
	skill_params = {'mu1': x_pam[0], 'sigma1': x_pam[1]}
	skill_bounds = [x_bounds[0], x_bounds[1]]

	workers = inputs.Input(var=x,
                           cdf=skill_cdf,
                       		params=skill_params,
                       		bounds=skill_bounds,
                      		)

	# define some default firms
	y, mu2, sigma2 = sym.var('y, mu2, sigma2')
	productivity_cdf = 0.5 + 0.5 * sym.erf((y - mu2) / sym.sqrt(2 * sigma2**2))
	productivity_params = {'mu2': y_pam[0], 'sigma2': y_pam[1]}
	productivity_bounds = [y_bounds[0], y_bounds[1]]

	firms = inputs.Input(var=y,
    	                 cdf=productivity_cdf,
        	             params=productivity_params,
            	         bounds=productivity_bounds,
                	     )
	""" 2. Unpack params and define the production function (MS) """
	# CES between x and y
	omega_A, sigma_A = sym.var('omega_A, sigma_A')
	A = ((omega_A * x**((sigma_A - 1) / sigma_A) + 
    	 (1 - omega_A) * y**((sigma_A - 1) / sigma_A))**(sigma_A / (sigma_A - 1))) 

	# Cobb-Douglas between l and r
	l, r, omega_B = sym.var('l, r, omega_B')
	B = l**omega_B * r**(1 - omega_B)

	F = A * B

	F_params = {'omega_A':params[0], 'omega_B':params[1], 'sigma_A':params[2]}

	""" 3. Solve the model """
	try:
		sol = Solve_Model(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess)		
	except AssertionError:
		try: 
			sol = Solve_Model(F, F_params, workers, firms, 'positive', 6000.0, 'vode', guess)
		except AssertionError:
			try: 
				sol = Solve_Model(F, F_params, workers, firms, 'positive', 6000.0, 'dopri5', guess)					 
			except AssertionError:
				try: 
					sol = Solve_Model(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess/100.0)
				except AssertionError:
					try: 
						sol = Solve_Model(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess*100.0)
					except AssertionError:
						print "OK JUST LEAVE IT", params
						return 400.00
	""" 4. Calculate and return """				 	
	mu_hat, theta_hat, w_hat = sol[0]
	guess = sol[1]	
	mse = Calculate_MSE(data, (mu_hat, theta_hat, w_hat) )
	print mse, params
	return mse