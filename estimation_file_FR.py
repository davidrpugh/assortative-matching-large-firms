from __future__ import division
import matplotlib.pyplot as plt
from scipy import stats, optimize
import numpy as np
import sympy as sym
import inputs
import models
import shooting
from estimation_class import *

est1 = HTWF_Estimation((0.0,1.0), (0.005, 20.0), (0.0,1.0), (0.005, 20.0), 142.85, 100000.0)
est1.InitializeFunction()
est1.import_data('SPA05.csv', ID=False, weights=False, logs=False, yearly_w=True)
ps = (0.6,0.7,0.99,1.0)
res = optimize.minimize(est1.StubbornObjectiveFunction, 
                        ps,args=(2000, 1e-3, 1e4), 
                        method='L-BFGS-B', jac=None, hess=None, hessp=None, 
                        bounds=((1e-3,1.0-1e-3),(1e-3,1.0-1e-3),(1e-3,1.0-1e-3),(0.5,2.0)), options={'eps':1e-2})