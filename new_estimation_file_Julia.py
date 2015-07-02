import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
import numpy as np
import sympy as sym
import csv

import inputs
import models
import shooting
import operator



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
	A tutple consisting on (theta(x), w(x), pis_from_model), final result from 

	"""
	flag_solver = False
	modelA = models.Model(assortativity=ass,
    	                 workers=workers,
        	             firms=firms,
            	         production=Fnc,
                	     params=F_params)

	solver = shooting.ShootingSolver(model=modelA)
	''' 1.Solve the Model '''
	solver.solve(ini, tol=tolerance, number_knots=N_knots, integrator=intg)
	
	''' 2.Check it is truly solved '''
	if not (solver._converged_firms(1e-6) and solver._converged_firms(1e-6)):
		flag_solver = True
	err_mesg = ("Fail to solve!")
   	assert (solver._converged_firms(1e-6) and solver._converged_firms(1e-6)), err_mesg

   	''' 3.Store vectors of results '''
	thetas = solver.solution['$\\theta(x)$'].values
	ws = solver.solution['$w(x)$'].values
	pis_fm = solver.solution['$\\pi(x)$'].values

	''' 4.Interpolate w(theta), pi(theta) '''
	w_theta = interp1d(thetas, ws)
	pi_theta = interp1d(thetas,pis_fm)
	
	return (w_theta, pi_theta, thetas), thetas[-1]


def import_data(file_name, ID=True, weights=False, logs=False):
    '''
    This function imports the data from a csv file, returns ndarrays with it

    Input
    -----

    file_name : (str) path and name of the csv file.
    ID (optional) : boolean, True if it contais ID in the first collunm
    weights : True if the data includes a weights column in the end
    logs: if the data is in logs (True) or in levels (False)

    Output
    ------
    Following np.ndarrays: profit (float), size (float), wage (float) (all in logs)

    '''
    # Opening data
    with open(file_name, 'rb') as f:
        reader = csv.reader(f)
       	data = list(reader)

    # Passing data to lists, then to arrays (should change this to make it all in one) 
    size = []
    wage = []
    profit = []
    wgts= []
    c = 0
    if ID==False:
    	c += 1
    for row in data[1:]:
        size.append(float(row[1-c]))
        wage.append(float(row[2-c]))
        profit.append(float(row[3-c]))
        if weights:
        	wgts.append(float(row[4-c]))

    if logs==False:
    	# Firm size in LOG workers (float)
    	size = np.log(np.array(size))
    	# Daily LOG average wage for each firm, in euros (float)
    	wage = np.log(np.array(wage))
    	# Declared LOG average profits for each firm per year, in euros (float)
    	profit = np.log(np.array(profit))
    else:
    	# Firm size in workers (int)
    	size = np.asarray(size)
    	# Daily average wage for each firm, in euros (float)
    	wage = np.asarray(wage)
    	# Declared average profits for each firm per year, in euros (float)
    	profit = np.asarray(profit)
    # In any case, weights should be the same
    wgts = np.array(wgts)

    if weights:
    	return size, wage, profit, wgts
    else:
    	return size, wage, profit

def pdf_workers(x, mu_workers=0.0, sigma_workers=1.0):
    	return np.sqrt(2)*np.exp(-(-mu_workers + np.log(x))**2/(2*sigma_workers**2))/(np.sqrt(np.pi)*x*sigma_workers)
    	
def Calculate_MSE(data, functions_from_model):
	'''
	Given some estimated shape distribution parameters, gives the mean square error related to
	the data's fitted distribution.

	Must have the module stats imported from scipy!!

	Parameters
    ----------
    data:			        (1x4) tuple of ndarrays. Data points to calculate the mse Order: thetas, wages, profits and weights if using.
	functions_from_model: 	(1x3) tuple of functions and data from the model solution. Order: theta(pi), w(pi), pis from model.
							Functions must be executable (that is, input a float and return a float)			

	Returns
    -------
	mse = Mean square errors of fit (tup of floats, one for each n)
	'''

	'''1. Unpack Data and functions'''
	pis = data[2]
	thetas = data[0]
	ws = data[1]
	if len(data)==4:
		weights = data[3]
	else
		weights = np.ones(len(pis))

	w_theta = functions_from_model[0]
	pi_theta = functions_from_model[1]
	thetas_from_model = functions_from_model[2]

	'''2. Calculate Mean Square error (REGRESSIONS)'''
	w_err = np.empty(0)
	pi_err = np.empty(0)

	for i in range(len(pis)):
		w_hat = np.log(w_theta(np.exp(thetas[i])))
		pi_hat = np.log(pi_theta(np.exp(thetas[i])))
		w_err = np.hstack((w_err, (w_hat-ws[i])**2*weights[i]))
		pi_err = np.hstack((pi_err, (pi_hat-pis[i])**2*weights[i]))

	mse_w = np.sum(w_err)
	mse_pi = np.sum(pi_err)

	'''3. Calculate Mean Square error (THETA DISTRIBUTION)'''
	#theta_KS = stats.ks_2samp(thetas, thetas_from_model)[0] # NEED TO WRITE
	cdf_theta_data = []
	r = 0.0
	for i in range(len(theta)):
    	r += thetas[i]*weights[i]
    	cdf_theta_data.append(r)  #calculates cdf of theta 

    cdf_theta_data = cdf_theta_data/cdf_theta_data[-1] #normalise to 1

    #getting size distribution from model - is theta_from_model the right variable here?
    n_thetas = dict(zip(list(map(str, range(0,1000))),thetas_from_model))
	sort_thetas = sorted(n_thetas.items(), key=operator.itemgetter(1))
	theta_range = sorted(thetas_from_model)


	pdf_x = pdf_workers(rxs)         # calculates pdf of xs in one step
	n_pdf_x = dict(enumerate(pdf_x)) # creates a dictionary where the keys are the #obs of x
	pdf_theta = np.empty(0)
	for pair in sort_thetas:
    	index = int(pair[0])
    	pdf_theta = np.hstack((pdf_theta,(n_pdf_x[index]/pair[1])))

	cdf_theta = np.cumsum(pdf_theta)
	cdf_theta = cdf_theta/cdf_theta[-1]
	cdf_theta_int = interp1d(np.log(theta_range),cdf_theta)

	theta_err = np.empty(0)

	for i in range(len(pis)):
		cdf_hat = cdf_theta_int(thetas[i])
		theta_err = np.hstack(theta_err, (cdf_hat-cdf_theta_data[i])**2)   #weighting not needed here because already in cdf

	mse_theta = np.sum(theta_err)


	return mse_theta + mse_pi + mse_w


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
	omega_A, sigma_A, Big_A = sym.var('omega_A, sigma_A, Big_A')
	A = ((omega_A * x**((sigma_A - 1) / sigma_A) + 
    	 (1 - omega_A) * y**((sigma_A - 1) / sigma_A))**(sigma_A / (sigma_A - 1))) 

	# Cobb-Douglas between l and r
	l, r, omega_B = sym.var('l, r, omega_B')
	B = l**omega_B * r**(1 - omega_B)

	F = Big_A * A * B

	F_params = {'omega_A':params[0], 'omega_B':params[1], 'sigma_A':params[2], 'Big_A':params[3]}

	""" 3. Solve the model """
	try:
		sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess)		
	except AssertionError:
		try: 
			sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'vode', guess)
		except AssertionError:
			try: 
				sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'dopri5', guess)					 
			except AssertionError:
				try: 
					sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess/100.0)
				except AssertionError:
					try: 
						sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess*100.0)
					except AssertionError, e:
						print "OK JUST LEAVE IT", params, "error:", e
						return 400000.00
	except ValueError, e:
		print "Wooops! ", params, e
		try:
			sol = Solve_Model2(F, F_params, workers, firms, 'negative', 6000.0, 'lsoda', guess)			
		except ValueError, e:
			print "OK JUST LEAVE IT", params, "error:", e
			return 4000000.00

	""" 4. Calculate and return """				 	
	theta_pi, w_pi, thetas = sol[0]
	guess = sol[1]	
	mse = Calculate_MSE(data, (theta_pi, w_pi, thetas) )
	print mse, params
	return mse




def StubbornObjectiveFunction2(params, data, x_pam, x_bounds, y_pam, y_bounds, guess):
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
	omega_A, sigma_A, Big_A = sym.var('omega_A, sigma_A, Big_A')
	A = ((omega_A * x**((sigma_A - 1) / sigma_A) + 
    	 (1 - omega_A) * y**((sigma_A - 1) / sigma_A))**(sigma_A / (sigma_A - 1))) 

	# Cobb-Douglas between l and r
	l, r, omega_B = sym.var('l, r, omega_B')
	B = l**omega_B * r**(1 - omega_B)

	F = Big_A * A * B

	F_params = {'omega_A':params[0], 'omega_B':params[1], 'sigma_A':params[2], 'Big_A':params[3]}

	""" 3. Solve the model """
	try:
		sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess)		
	except AssertionError:
		try: 
			sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'vode', guess)
		except AssertionError:
			try: 
				sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'dopri5', guess)					 
			except AssertionError:
				try: 
					sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess/100.0)
				except AssertionError:
					try: 
						sol = Solve_Model2(F, F_params, workers, firms, 'positive', 6000.0, 'lsoda', guess*100.0)
					except AssertionError, e:
						print "OK JUST LEAVE IT", params, "error:", e
						return 400000.00
	except ValueError, e:
		print "Wooops! ", params, e
		try:
			sol = Solve_Model2(F, F_params, workers, firms, 'negative', 6000.0, 'lsoda', guess)			
		except ValueError, e:
			print "OK JUST LEAVE IT", params, "error:", e
			return 4000000.00

	""" 4. Calculate and return """				 	
	theta_pi, w_pi, thetas = sol[0]
	guess = sol[1]	
	mse = Calculate_MSE(data, (theta_pi, w_pi, thetas) )
	print mse, params
	return mse