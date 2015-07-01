"""
Estimates a model using forward shooting.

@authors : Cristina & Julia & David R. Pugh

"""
from __future__ import division

from scipy import stats
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import numpy as np
import sympy as sym
import csv

import inputs
import models
import shooting
import operator

class HTWF_Estimation(object):
	'''
	Call the Instructions() function for help.
	'''
	def __init__(self, x_pam, x_bounds, y_pam, y_bounds, x_scaling,A_scaling):
		'''
		Puts together an Heterogeneous Workers and Firms model with the parameters especified,
		imports data to carry on the estimation and estimates the parameters of the model:
		                       { omega_A, omega_B, sigma and Big_A }

		Call the Instructions() function to print the condensed user manual.

		Parameters:
		-------

		x_pam: tuple or list, with floats mu1 and sigma1.
		x_bounds: tuple or list, with floats x.lower and x.upper.
		y_pam: tuple or list, with floats mu2 and sigma2.
		y_bounds: tuple or list, with y.lower and y.upper.
		x_scaling: float, average workers per firm, to scale up the size of x.
				   May change in the future to make it endogenous.
		A_scaling: float, scaling parameter for Big_A. To be chosen from the data.
		
		'''

		self.x_pam = x_pam
		self.x_bounds = x_bounds
		self.y_pam = y_pam
		self.y_bounds = y_bounds
		self.x_scaling = x_scaling
		self.A_scaling = A_scaling 
		
		self.F = None
		self.workers = None
		self.firms = None

		self.data = None

		self.ready = False	

		self.current_sol = None
		self.last_results = np.zeros(8)

		self.error = 4000000.0

	def Instructions(self):
		'''
		Prints the instructions of the HTWF_Estimation class.

		'''
		inst1 = 'Step 1. Call InitializeFunction() to get the production function stored.'
		inst2 = "Step 2. Call import_data('datafile') to get the data stored."
		inst2b = "      (if you need the wages to be annulized, when creating" 
		inst2c = "	     the HTWF_Estimation instance add yearly_w=True at the end)"
		inst3 = "Step 3. You are ready to go! Call StubbornObjectiveFunction with your initial set of parameters (4),"
		inst3b = "       number of grid points, tolerance, and a big initial guess for the firm)"
		for ins in [inst1,inst2,inst2b,inst2c,inst3,inst3b]:
			print ins

	def InitializeFunction(self):
		'''
		Resets workers, firms, production function given the parameters especified when creating the instance.

		Needs to be executed before solving the model.

		'''
		# define some default workers skill
		x, mu1, sigma1 = sym.var('x, mu1, sigma1')
		skill_cdf = self.x_scaling*(0.5 + 0.5 * sym.erf((sym.log(x) - mu1) / sym.sqrt(2 * sigma1**2)))
		skill_params = {'mu1': self.x_pam[0], 'sigma1': self.x_pam[1]}
		skill_bounds = [self.x_bounds[0], self.x_bounds[1]]

		workers = inputs.Input(var=x,
                           cdf=skill_cdf,
                       		params=skill_params,
                       		bounds=skill_bounds,
                      		)

		# define some default firms
		y, mu2, sigma2 = sym.var('y, mu2, sigma2')
		productivity_cdf = 0.5 + 0.5 * sym.erf((sym.log(y) - mu2) / sym.sqrt(2 * sigma2**2))
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

		F = (Big_A*self.A_scaling) * A * B

		self.workers = workers
		self.firms = firms 
		self.F = F


	def import_data(self,file_name, ID=True, weights=False, logs=False,yearly_w=False, change_weight=False):
	    '''
	    This function imports the data from a csv file, returns ndarrays with it.

	    It can accomodate for having IDs in the first column, for data not in logs already,
	 	for weights (normalized or not) and can transform daily to yearly wages.

	    Parameters
	    -----

	    file_name : (str) path and name of the csv file.
	    ID (optional) : boolean, True if it contais ID in the first collunm
	    weights : True if the data includes a weights column in the end
	    logs: if the data is in logs (True) or in levels (False)
	    yearly_w: (optional) boolean, True if the wages need to be "annualized". 
		change_weight: (optional) boolean, True if data weights need to be "normalized". 

	    Returns
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
	    	size = np.array(size)
	    	# Daily average wage for each firm, in euros (float)
	    	wage = np.array(wage)
	    	# Declared average profits for each firm per year, in euros (float)
	    	profit = np.array(profit)
	    # In any case, weights should be the same
	    wgts = np.array(wgts)

	    if yearly_w:
	    	wage = np.log(np.exp(wage)*360) # ANNUAL wage

	    if change_weight:
	    	wgts = wgts/np.sum(wgts)*len(wgts)

	    # Storing results
	    if weights:
	    	self.data = (size, wage, profit, wgts)
	    else:
	    	self.data = (size, wage, profit)

	def pdf_workers(self,x):
		'''
		For a given x returns the corresponding pdf value, 
		according to the paramenters especified when creating the instance.

		Parameters:
		-----------

		x: (float or int) a point in x (the distribution of firm size)

		Returns:
		--------

		pdf(x) (float) according to the parameters of the instance.

		'''
	    	return np.sqrt(2)*np.exp(-(-self.x_pam[0] + np.log(x))**2/(2*self.x_pam[1]**2))/(np.sqrt(np.pi)*x*self.x_pam[1])

	def Solve_Model(self, F_params, N_knots, knots, intg, ini, tolerance):
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
		A tutple consisting on: 
			(theta(w) (scipy.interpolate.interpolate.interp1d), 
			 theta(pi) (scipy.interpolate.interpolate.interp1d), 
			 thetas from the model (np.array), 
			 xs from the model(np.array))

		"""
		Fnc=self.F
		workers= self.workers 
		firms=self.firms

		modelA = models.Model(assortativity='positive',
	    	                 workers=workers,
	        	             firms=firms,
	            	         production=Fnc,
	                	     params=F_params)

		solver = shooting.ShootingSolver(model=modelA)
		''' 1.Solve the Model '''
		solver.solve(ini, tol=tolerance, number_knots=N_knots, knots=knots, integrator=intg)
		
		''' 2.Check it is truly solved '''
		err_mesg = ("Fail to solve!")
	   	assert (solver._converged_firms(tolerance) and solver._converged_workers(tolerance)), err_mesg

	   	''' 3.Store vectors of results '''
		thetas = solver.solution['$\\theta(x)$'].values
		ws = solver.solution['$w(x)$'].values
		pis_fm = solver.solution['$\\pi(x)$'].values
		xs_fm = solver.solution.index.values
		self.current_sol = solver.solution

		''' 4.Interpolate w(theta), pi(theta) '''
		w_theta = interp1d(ws, thetas,bounds_error=False,fill_value=-99.0)
		pi_theta = interp1d(pis_fm,thetas,bounds_error=False,fill_value=-99.0)
		
		return (w_theta, pi_theta, thetas, xs_fm)

	def Calculate_MSE(self, functions_from_model,penalty=100):
		'''
		Given some estimated shape distribution parameters, gives the mean square error related to
		the data's fitted distribution.

		Must have the data imported!

		Parameters
	    ----------
	    data:			        (1x4) tuple of ndarrays. Data points to calculate the mse Order: thetas, wages, profits and weights if using.
		functions_from_model: 	(1x3) tuple of functions and data from the model solution. Order: theta(pi), w(pi), thetas from model, xs from model.
								Functions must be executable (that is, input a float and return a float)
		penalty :				Penalty for each observation out of the model range that appears in the data (int or float)		

		Returns
	    -------
		Mean square errors of fit (float), array with the three erros separately (1x3 np.array)

		Order: firm distribution error, wage fit error, profit fit error

		'''
		# Check data is in!
		err_mesg = ("Need to import data first!")
		assert self.data != None, err_mesg

		# 1. Unpack Data and functions
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

		# 2. Calculate Mean Square error (REGRESSIONS)
		w_err = np.empty(0)
		pi_err = np.empty(0)

		for i in range(len(pis)):
			# For each wage observation out of range, penalty
			if theta_w(np.exp(ws[i]))==-99.0:
				#print 'w out of range'
				w_err = np.hstack((w_err,penalty*weights[i]))	
			else:
				w_hat = np.log(theta_w(np.exp(ws[i])))
				w_err = np.hstack((w_err, (w_hat-thetas[i])**2*weights[i]))
			# For each profit observation out of range, penalty
			if theta_pi(np.exp(pis[i]))==-99.0:
				#print 'pi out of range'			
				pi_err = np.hstack((pi_err,penalty*weights[i]))
			else:
				pi_hat = np.log(theta_pi(np.exp(pis[i])))
				pi_err = np.hstack((pi_err, (pi_hat-thetas[i])**2*weights[i]))
		# Adding up the errors
		mse_w = np.sum(w_err)
		mse_pi = np.sum(pi_err)

		# 3. Calculate Mean Square error (THETA DISTRIBUTION)
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
		pdf_x = self.pdf_workers(xs_from_model)        	# calculates pdf of xs in one step
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
				theta_err = np.hstack((theta_err, penalty*weights[i]))
			else:
				theta_err = np.hstack((theta_err, (cdf_hat-cdf_theta_data[i])**2))   #weighting not needed here because already in cdf

		mse_theta = np.sum(theta_err)

		return mse_theta + mse_pi + mse_w, np.array([mse_theta, mse_w, mse_pi])

	def Areureadyforthis(self):
		'''
		Test if you can go ahead with estimation: you have firms, workers, production function and data ready. (Need to add some more refined tests)

		Returns
		------
		True (bool) if it has everything it needs to carry on to the estimation.

		'''
		if self.F != None and self.workers != None and self.firms != None and self.data != None:
			self.ready = True

	def StubbornObjectiveFunction(self, params, N_points, grid, tol_i, guess):
		"""
		Calculates the sum of squared errors from the model with given parameters.

		Assumes lognormal distribution for both firms and workers, with parameters especified when creating the instance.

		Parameters:
		-----------
			(1x4) tuple containing: (omega_A, omega_B, sigma, Big_A) (all of them floats)
			grid_points: number of points in the grid of xs (float or int)
			tol_i: tolerance for the solver (float) 
			guess: initial guess for upper firm size (float or int) 


		Returns:
		--------
			Sum of squared residuals of theta(ws), theta(pis), cdf(theta).

		"""
		# 1. Check if F and data are stored correctly
		self.Areureadyforthis()
		err_mesg = 'You ar enot ready yet to go! Check you have imputed F and imported the data!'
		assert self.ready

		# 2. Unpack params
		F_params = {'omega_A':params[0], 'omega_B':params[1], 'sigma_A':params[2], 'Big_A':params[3]}

		# 3. Solve the model
		try:
			sol = self.Solve_Model(F_params, N_points, grid, 'lsoda', guess, tol_i)		
		except AssertionError ,e:
			if e=='Failure! Need to increase initial guess for upper bound on firm size!':
				guess = guess * 10
				print guess
			try: 
				sol = self.Solve_Model(F_params, N_points, grid, 'vode', guess, tol_i)
			except AssertionError:
				try: 
					sol = self.Solve_Model(F_params, N_points, grid, 'lsoda', guess*100.0, tol_i)					 
				except AssertionError:
					try: 
						sol = self.Solve_Model(F_params, N_points, grid, 'lsoda', guess/100.0, tol_i)
					except AssertionError:
						try:
							sol = self.Solve_Model(F_params, N_points, grid, 'lsoda', guess, tol_i*10)
						except AssertionError:
							if N_points is not None:
								try: 
									sol = self.Solve_Model(F_params, N_points*1.5, grid,'lsoda', guess, tol_i)
								except AssertionError, e:
									print "K.", params, "error:", e
									self.error = self.error*1.2
									return self.error
							else:
								print "K.", params, "error:", e
								self.error = self.error*1.2
								return self.error
				except ValueError, e:
					print "Wrong assortativity ", params, e
					self.error = self.error*1.2
					return self.error
			except ValueError, e:
				print "Wrong assortativity ", params, e
				self.error = self.error*1.2
				return self.error
		except ValueError, e:
			print "Wrong assortativity ", params, e
			self.error = self.error*1.2
			return self.error


		# 4. Calculate and return				 	
		functions_model = sol
		output = self.Calculate_MSE(functions_model)
		mse = output[0]
		sol_path = np.hstack((np.array(params),output[1], mse))
		self.last_results = np.vstack((self.last_results,sol_path))
		print mse, params
		return mse

	def write_solution_path(self):
		'''
		Stores the solution path over multiple calls of StubbornObjectiveFunction in a csv file.

		Returns:
		--------
		A csv file called 'results' with the path from the solver.
		The rder of columns is:

		omega_A, omega_B, sigma, Big_A, firm size distribution error, wages error, profits error, total error

		The first row is just zeros.
		'''
		self.last_results.tofile('results.csv',sep=',',format='%10.5f')

	def reset_solution_path(self):
		'''
		Resets the minimizer solution path in self.last_results. To be called after a minimization routine to clean cache.
		'''
		self.last_results = np.zeros(8)

	def get_cdf(self,thetas_fm,xs_fm):
		'''
		Gets the cdf value of a given firm size, by interpolating the implied distribution from the results of solving the model.

		Parameters:
		-----------
		thetas_fm: thetas from model (np.array or similar)
		xs_fm: xs grid points from model (np.array or similar)

		Returns:
		--------
		An executable numeric function, that omits values out of range (scipy.interpolate.interpolate.interp1d).

		'''
	
		n_thetas = dict(zip(list(map(str, range(0,len(thetas_fm)))),thetas_fm))
		sort_thetas = sorted(n_thetas.items(), key=operator.itemgetter(1))
		theta_range = sorted(thetas_fm)

		# Using the pdf of workers
		pdf_x = self.pdf_workers(xs_fm)        	# calculates pdf of xs in one step
		n_pdf_x = dict(enumerate(pdf_x)) 			# creates a dictionary where the keys are the #obs of x
		pdf_theta_hat = np.empty(0)
		for pair in sort_thetas:
			index = int(pair[0])
			pdf_theta_hat  = np.hstack((pdf_theta_hat ,(n_pdf_x[index]/pair[1])))

		cdf_theta_hat  = np.cumsum(pdf_theta_hat )			# Backing up model cdf
		cdf_theta_hat  = cdf_theta_hat /cdf_theta_hat[-1] 	# Normilization of the model cdf
		cdf_theta_int = interp1d(np.log(theta_range),cdf_theta_hat,bounds_error=False)
	
		return cdf_theta_int

	def Plot_data(self):
		'''
		Plots the data stored: 
			subplot 1: wages vs sizes, size(wages) from the model
			subplot 2: profits vs sizes, size(wages) from the model
			subplot 3: cdf(sizes) from the data (green), cdf(sizes) from the model (blue)
		
		Returns:
		--------
		pyplot plot with the three subplots as specified above.

		'''
		# Check data is in!
		err_mesg = ("Need to import data first!")
		assert self.data != None, err_mesg
		# Uncompress data
		if len(self.data)==3:
			theta, wage, profit = self.data
		else:
			theta, wage, profit, weights = self.data
			
		cdf_theta_data = []
		r = 0.0
		for i in range(len(theta)):
		    r += theta[i]
		    cdf_theta_data.append(r)
		cdf_theta_data = np.array(cdf_theta_data)/cdf_theta_data[-1]

		plt.figure(figsize=(15,5))
		plt.suptitle('Data Plot', fontsize=20)
		plt.subplot(131)
		plt.scatter(wage,theta, marker='x')
		plt.xlabel('$w$', fontsize=18)
		plt.ylabel('$\\theta$', fontsize=18)

		plt.subplot(132)
		plt.scatter(profit,theta, marker='x', color='r')
		plt.xlabel('$\\pi$', fontsize=18)
		plt.ylabel('$\\theta$', fontsize=18)

		plt.subplot(133)
		plt.plot(theta, cdf_theta_data, color='g')
		plt.ylabel('$cdf(\\theta)$', fontsize=18)
		plt.xlabel('$\\theta$', fontsize=18)

		plt.show()

	def Plot_solution(self):
		'''
		Plots the data stored and the functions obtained from the last estimation:
		 	subplot 1: wages vs sizes, size(wages) from the model
			subplot 2: profits vs sizes, size(wages) from the model
			subplot 3: cdf(sizes) from the data (green), cdf(sizes) from the model (blue)

		Returns:
		--------
		pyplot plot with the three subplots as specified above.

		'''
		# Check data is in!
		err_mesg = ("Need to import data first!")
		assert self.data != None, err_mesg
		# Check the model is solved!
		err_mesg = ("Need to solve the model first!")
		assert type(self.current_sol) != None, err_mesg
		# Uncompress data
		theta, wage, profit = self.data
		cdf_theta_data = []
		r = 0.0
		for i in range(len(theta)):
		    r += theta[i]
		    cdf_theta_data.append(r)
		cdf_theta_data = np.array(cdf_theta_data)/cdf_theta_data[-1]

		plthetas = self.current_sol['$\\theta(x)$'].values
		plws = self.current_sol['$w(x)$'].values
		plpis = self.current_sol['$\\pi(x)$'].values
		plxs = self.current_sol.index.values
				
		''' 4.Interpolate w(theta), pi(theta) '''
		th_w = interp1d(plws, plthetas,bounds_error=False)
		th_pi = interp1d(plpis,plthetas,bounds_error=False)
		cdf_model = self.get_cdf(plthetas,plxs)

		plt.figure(figsize=(15,5))
		plt.suptitle('Best Fit of the day', fontsize=20)
		plt.subplot(131)
		plt.scatter(wage,theta, marker='x')
		plt.plot(sorted(wage),np.log(th_w(np.exp(sorted(wage)))))
		plt.xlabel('$w$', fontsize=18)
		plt.ylabel('$\\theta$', fontsize=18)

		plt.subplot(132)
		plt.scatter(profit,theta, marker='x', color='r')
		plt.plot(sorted(profit),np.log(th_pi(np.exp(sorted(profit)))))
		plt.xlabel('$\\pi$', fontsize=18)
		plt.ylabel('$\\theta$', fontsize=18)

		plt.subplot(133)
		plt.plot(theta, cdf_theta_data, color='g')
		plt.plot(theta,cdf_model(theta))
		plt.ylabel('$cdf(\\theta)$', fontsize=18)
		plt.xlabel('$\\theta$', fontsize=18)

		plt.show()

