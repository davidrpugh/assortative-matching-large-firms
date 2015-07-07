from __future__ import division
import matplotlib.pyplot as plt
from scipy import stats, optimize
import numpy as np
import sympy as sym
import inputs
import models
import shooting
from estimation_class2 import *

knots= np.logspace(np.log(0.005), np.log(200.0), 1000, endpoint=True, base=np.e)
est1 = HTWF_Estimation((0.0,1.0), (0.005, 200.0), (0.0,1.0), (0.005, 200.0), 14.0, 100000.0,knots)
est1.InitializeFunction()
est1.import_data('dummysol.csv', ID=False, weights=False, logs=False, yearly_w=False,dummy=True,labels=False)
ps = (0.04,0.4,0.8,5.0)
res = optimize.minimize(est1.StubbornObjectiveFunction, 
                        ps,args=(None, 1e-4, 14500.0), 
                        method='L-BFGS-B', jac=None, hess=None, hessp=None, 
                        bounds=((1e-2,1.0-1e-2),(1e-2,1.0-1e-2),(1e-2,1.0-1e-2),(1e-2,None)), options={'eps':1e-2})
print res
