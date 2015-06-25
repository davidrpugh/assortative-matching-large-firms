"""
Estimates a model using forward shooting.

@authors : Cristina & Julia & David R. Pugh

"""
from __future__ import division

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

class HTWF_Estimation(object):

	def __init__(self, x_pam, x_bounds, y_pam, y_bounds, x_scaling):

		self.x_pam = x_pam
		self.x_bounds = x_bounds
		self.y_pam = y_pam
		self.y_bounds = y_bounds
		self.x_scaling = x_scaling 
		
		self.F = None
		self.workers = None
		self.firms = None

		self.data = None

		self.ready = False

	def InitializeFunction(self):
		'''
		Resets workers, firms, production function
		'''
		# define some default workers skill
		x, mu1, sigma1 = sym.var('x, mu1, sigma1')
		skill_cdf = self.x_scaling*0.5 + 0.5 * sym.erf((x - mu1) / sym.sqrt(2 * sigma1**2))
		skill_params = {'mu1': self.x_pam[0], 'sigma1': self.x_pam[1]}
		skill_bounds = [self.x_bounds[0], self.x_bounds[1]]

		workers = inputs.Input(var=x,
                           cdf=skill_cdf,
                       		params=skill_params,
                       		bounds=skill_bounds,
                      		)

		# define some default firms
		y, mu2, sigma2 = sym.var('y, mu2, sigma2')
		productivity_cdf = 0.5 + 0.5 * sym.erf((y - mu2) / sym.sqrt(2 * sigma2**2))
		productivity_params = {'mu2': self.y_pam[0], 'sigma2': self.y_pam[1]}
		productivity_bounds = [self.y_bounds[0], self.y_bounds[1]]

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

		self.workers = workers
		self.firms = firms 
		self.F = F


	def import_data(self,file_name, ID=True, weights=False, logs=False):
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
	    	self.data = (size, wage, profit, wgts)
	    else:
	    	self.data = (size, wage, profit)

	def pdf_workers(self,x):
		'''
		For a given x returns the corresponding pdf value.
		'''
	    	return np.sqrt(2)*np.exp(-(-self.x_pam[0] + np.log(x))**2/(2*self.x_pam[1]**2))/(np.sqrt(np.pi)*x*self.x_pam[1])

	def Solve_Model(self, F_params, N_knots, intg, ini, tolerance):
		"""
		Function that solves the sorting model and returns functions mu(x), theta(x), w(x).

		Assumes monotonicity,log normal distributions for size and wages.
		To fit the data, it uses worker skill as main variable.
		It also only uses the shooting solver, checking for completition.

		Parameters
	    ----------
		F_params: Parameters of Fnc (dic)
		N_knots: number of knots (int)
		intg: a string with the integrator of preference (string)
		ini: initial guess (float)
		tolerance: tolerance for the solver (float)
		ass: assortativity ('positive' or 'negative') (string)

		Returns
	    -------
		A tutple consisting on (theta(x), w(x), pis_from_model), final result from 

		"""
		Fnc=self.F
		workers= self.workers 
		firms=self.firms

		flag_solver = False
		modelA = models.Model(assortativity='positive',
	    	                 workers=workers,
	        	             firms=firms,
	            	         production=Fnc,
	                	     params=F_params)

		solver = shooting.ShootingSolver(model=modelA)
		''' 1.Solve the Model '''
		solver.solve(ini, tol=tolerance, number_knots=N_knots, integrator=intg)
		
		''' 2.Check it is truly solved '''
		if not (solver._converged_firms(tolerance) and solver._converged_firms(tolerance)):
			flag_solver = True
		err_mesg = ("Fail to solve!")
	   	assert (solver._converged_firms(tolerance) and solver._converged_firms(tolerance)), err_mesg

	   	''' 3.Store vectors of results '''
		thetas = solver.solution['$\\theta(x)$'].values
		ws = solver.solution['$w(x)$'].values
		pis_fm = solver.solution['$\\pi(x)$'].values
		xs_fm = solver.solution.index.values

		''' 4.Interpolate w(theta), pi(theta) '''
		w_theta = interp1d(ws, thetas,bounds_error=False,fill_value=-99.0)
		pi_theta = interp1d(pis_fm,thetas,bounds_error=False,fill_value=-99.0)
		
		return (w_theta, pi_theta, thetas, xs_fm)

	def Calculate_MSE(self, functions_from_model,penalty=100):
		'''
		Given some estimated shape distribution parameters, gives the mean square error related to
		the data's fitted distribution.

		Must have the module stats imported from scipy!!

		Parameters
	    ----------
	    data:			        (1x4) tuple of ndarrays. Data points to calculate the mse Order: thetas, wages, profits and weights if using.
		functions_from_model: 	(1x3) tuple of functions and data from the model solution. Order: theta(pi), w(pi), thetas from model, xs from model.
								Functions must be executable (that is, input a float and return a float)
		penalty :				Penalty for each observation out of the model range that appears in the data			

		Returns
	    -------
		mse = Mean square errors of fit (tup of floats, one for each n)
		'''

		'''1. Unpack Data and functions'''
		data = self.data

		pis = data[2]
		thetas = data[0]
		ws = data[1]
		if len(data)==4:
			weights = data[3]
		else:
			weights = np.ones(len(pis))

		theta_w = functions_from_model[0]
		theta_pi = functions_from_model[1]
		thetas_from_model = functions_from_model[2]
		xs_from_model = functions_from_model[3]

		'''2. Calculate Mean Square error (REGRESSIONS)'''
		w_err = np.empty(0)
		pi_err = np.empty(0)

		for i in range(len(pis)):
			# For each wage observation out of range, penalty
			if theta_w(np.exp(ws[i]))==-99.0:
				#print 'w out of range'
				w_err = np.hstack((w_err,penalty))	
			else:
				w_hat = np.log(theta_w(np.exp(ws[i])))
				w_err = np.hstack((w_err, (w_hat-thetas[i])**2*weights[i]))
			# For each profit observation out of range, penalty
			if theta_pi(np.exp(pis[i]))==-99.0:
				#print 'pi out of range'			
				pi_err = np.hstack((pi_err,penalty))
			else:
				pi_hat = np.log(theta_pi(np.exp(pis[i])))
				pi_err = np.hstack((pi_err, (pi_hat-thetas[i])**2*weights[i]))
		# Adding up the errors
		mse_w = np.sum(w_err)
		mse_pi = np.sum(pi_err)

		'''3. Calculate Mean Square error (THETA DISTRIBUTION)'''
		# Getting cdf from data
		cdf_theta_data = []
		r = 0.0
		for i in range(len(thetas)):
			r += thetas[i]*weights[i]
			cdf_theta_data.append(r)  #calculates cdf of theta 

		cdf_theta_data = np.array(cdf_theta_data)/cdf_theta_data[-1] #normalise to 1

	    # getting size distribution from model
	    # Sorting the thetas
		n_thetas = dict(zip(list(map(str, range(0,len(thetas_from_model)))),thetas_from_model))
		sort_thetas = sorted(n_thetas.items(), key=operator.itemgetter(1))
		theta_range = sorted(thetas_from_model)

		# Using the pdf of workers
		pdf_x = pdf_workers(xs_from_model)        	# calculates pdf of xs in one step
		n_pdf_x = dict(enumerate(pdf_x)) 			# creates a dictionary where the keys are the #obs of x
		pdf_theta_hat = np.empty(0)
		for pair in sort_thetas:
			index = int(pair[0])
			pdf_theta_hat  = np.hstack((pdf_theta_hat ,(n_pdf_x[index]/pair[1])))

		cdf_theta_hat  = np.cumsum(pdf_theta_hat )			# Backing up model cdf
		cdf_theta_hat  = cdf_theta_hat /cdf_theta_hat [-1] 	# Normilization of the model cdf
		cdf_theta_int = interp1d(np.log(theta_range),cdf_theta_hat,bounds_error=False,fill_value=-99.0)

		# Calculate the error
		theta_err = np.empty(0)
		for i in range(len(pis)):
			cdf_hat = cdf_theta_int(thetas[i])
			if cdf_hat == -99.0:
				theta_err = np.hstack((theta_err, penalty))
			else:
				theta_err = np.hstack((theta_err, (cdf_hat-cdf_theta_data[i])**2))   #weighting not needed here because already in cdf

		mse_theta = np.sum(theta_err)

		print 'errors:',mse_theta, mse_pi, mse_w

		return mse_theta + mse_pi + mse_w

	def Areureadyforthis(self):
		'''
		(Need to add some more refined tests)
		'''
		if self.F != None and self.workers != None and self.firms != None and self.data != None:
			self.ready = True

	def StubbornObjectiveFunction(self, params, grid_points, tol_i, guess):
		"""
		Calculates the sum of squared errors from the model with given parameters.

		Assumes lognormal distribution for both firms and workers.

		In: (1x4) tuple containing: (omega_A, omega_B, sigma, Big_A)
			grid_points: float or int
			tol_i: (float) tolerance
			guess: (float or int) initial guess


		Out: Sum of squared residuals

		"""
		# 1. Check if F and data are stored correctly
		self.Areureadyforthis()
		err_mesg = 'You ar enot ready yet to go! Check you have imputed F and imported the data!'
		assert self.ready

		# 2. Unpack params
		F_params = {'omega_A':params[0], 'omega_B':params[1], 'sigma_A':params[2], 'Big_A':params[3]}

		# 3. Solve the model
		try:
			sol = self.Solve_Model(F_params, grid_points, 'lsoda', guess, tol_i)		
		except AssertionError:
			try: 
				sol = self.Solve_Model(F_params, grid_points, 'vode', guess, tol_i)
			except AssertionError:
				try: 
					sol = self.Solve_Model(F_params, grid_points, 'lsoda', guess*100.0, tol_i)					 
				except AssertionError:
					try: 
						sol = self.Solve_Model(F_params, grid_points, 'lsoda', guess/100.0, tol_i)
					except AssertionError:
						try:
							sol = self.Solve_Model(F_params, grid_points, 'lsoda', guess, tol_i*10)
						except AssertionError:
							try: 
								sol = self.Solve_Model(F_params, grid_points*1.5, 'lsoda', guess, tol_i)
							except AssertionError, e:
								print "OK JUST LEAVE IT", params, "error:", e
								return 400000.00

		# 4. Calculate and return			 	
		functions_model = sol
		mse = self.Calculate_MSE(self.data, functions_model)
		print mse, params
		return mse


