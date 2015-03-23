import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import PchipInterpolator
import numpy as np
import sympy as sym
import csv

import inputs
import models
import shooting



def Solve_Model(Fnc, F_params, workers, firms, ass, N_knots, intg, ini):
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

	''' 4.Interpolate mu(x), theta(x), w(x) '''
	mu_x = PchipInterpolator(x, mus)
	theta_x = PchipInterpolator(x, thetas)
	w_x = PchipInterpolator(x, ws)

	
	return (mu_x, theta_x, w_x), thetas[-1]

def import_data(file_name):
    '''
    This function imports the data from a csv file, returns ndarrays with it

    Input
    -----

    file_name : (str) path and name of the csv file.

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
    firm_age = []
    industry_code = []
    region = []
    for row in data[1:]:
        size.append(int(row[1]))
        wage.append(float(row[2]))
        profit.append(float(row[3]))
        skill_w.append(float(row[4]))
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

	return (np.mean(mu_err), np.mean(theta_err), np.mean(w_err))



