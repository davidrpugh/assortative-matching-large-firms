#
# This file finds the optimal coefficients of a polynomial approximation to the discontinous function k*(x,y,l,r)
# and then writes the resulting sympy expressions down in k_params.txt file.
#
# To be used in conjunction with the solver of the matching model.
#


from __future__ import division
import numpy as np
import sympy as sym
from scipy.optimize import leastsq

## Setting the parameters of the model (calibrated values)
R, rho, gamma, eta, L, A, kapa = sym.var('R, rho, gamma, eta, L, A, kapa')
### Defining variables ###
k, x, y, l, r = sym.var('k, x, y, l, r')

## Defining auxiliary functions a, c, d ##
a = -2*R
c = r*A*kapa*(eta**2)
d = r*A*kapa*eta*(1-eta)*(y*(l/r)**x)**(0.25)
K = sym.var('K')

## Defining the polynomial to solve ##
pol_K = K**3*a + K*c + d
polik = sym.lambdify((K,x,y,l,r,A,kapa,eta,R),pol_K)

## Defining auxiliary functions p, q ##

p = -(eta**2*A*r*kapa)/(2*R)
q = -(eta*(1-eta)*kapa*r*A*(y*(l/r)**x)**(1/4))/(2*R)

## The Roots: Hyperbolic approach ##

# CASE 1: In case p<0 and 4p^3+27q^2>0 we have one root:
hyper_root = 2 * (-p/3)**(0.5) * sym.cosh( (1/3)*sym.acosh( ((3*q)/(2*p)) * (-3/p)**(0.5) ) )

# CASE 2: In case p>0, we also have one root:
hyper_root2 = -2* (p/3)**(0.5) * sym.sinh( (1/3)*sym.asinh( ((3*q)/(2*p)) * (3/p)**(0.5) ) )

# CASE 3: In case p<0 and 4p^3+27q^2<=0 we have three real roots, of which the biggest is:
root_zero = (2*(-p/3)**(0.5))*sym.cos( (1/3)*sym.acos( ((3*q)/(2*p)) * (-3/p)**(1/2) ) )

# Lambdifying the roots ##
HRt = sym.lambdify((x,y,l,r,A,kapa,eta,R), hyper_root)
HRt2 = sym.lambdify((x,y,l,r,A,kapa,eta,R), hyper_root2)
Rt0 = sym.lambdify((x,y,l,r,A,kapa,eta,R), root_zero)

## Trigonometric condition function ##

def trig_condition(x,y,l,r,A,kappa,eta,R):
    '''
    Condition for three real roots
    
    Input: A series of model parameters and values (x,y,l,r,A,kappa,eta,R)
    
    Output: An interger for the case in which we are in (see above)
    '''
    p = -(eta**2*A*r*kappa)/(2*R)
    q = -(eta*(1-eta)*kappa*r*A*(y*(l/r)**x)**(1/4))/(2*R)
    pq = 4*p**3+27*q**2
    if p<0 and pq<=0:
        return 3
    elif p<0 and pq>0:
        return 1
    elif p>0:
        return 2


## Integrated function for case 1 and 3 ##

def My_k_star(x,y,l,r,A,kappa,eta,R):
    '''
    Function that returns the root that applies 
    depending on the parameters we have (input).
    
    Input: A series of model parameters and values (x,y,l,r,A,kappa,eta,R)
    
    Output: Tuple with
            - a float with the value of the root.
            - an interger with the case.
    '''
    if trig_condition(x,y,l,r,A,kappa,eta,R) == 1:
        return (HRt(x,y,l,r,A,kappa,eta,R),1)
    elif trig_condition(x,y,l,r,A,kappa,eta,R) == 3:
        return (Rt0(x,y,l,r,A,kappa,eta,R),3)

print "My_kstar done"

## Creating the Grid ##

y3d = np.logspace(-2,1.0,100)
l3d = np.logspace(-2,3.0,100)
x3d = np.logspace(-2,1.0,100)

k_array = []
k_array_rich = []
x_array = []
y_array = []
l_array = []

for x in x3d:
    for y in y3d:       
        for l in l3d:
            k_array.append((My_k_star(x,y,l,1.0,0.5105,1.0,0.89,0.83684)[0])**4) # WARNING: APPENDING FINAL K
            k_array_rich.append(((My_k_star(x,y,l,1.0,1.0,1.0,0.89,0.13099)[0])**4))
            x_array.append(x)
            y_array.append(y)
            l_array.append(l)

k_array = np.array(k_array)
k_array_rich = np.array(k_array_rich)
x_array = np.array(x_array)
y_array = np.array(y_array)
l_array = np.array(l_array)

print "Grid done"

## Solving the function approximation problem ##

def res_ten3(params, kr, z, a, b):
    z1,z2,a1,a2,b1,b2,za,zb,ab,z2a,za2,z2a2,z2b,zb2,z2b2,a2b,ab2,a2b2,zab,z2ab,za2b,zab2,z2a2b,z2ab2,za2b2,z2a2b2 = params
    return z1*z + z2*z**2 + a1*a + a2*a**2 + b1*b + b2*b**2 + za*z*a + zb*z*b + ab*a*b + z2a*z**2*a + za2*z*a**2 + z2a2*z**2*a**2 + z2b*z**2*b + zb2*z*b**2 + z2b2*z**2*b**2 + a2b*a**2*b + ab2*a*b**2 + a2b2*a**2*b**2 + zab*z*a*b + z2ab*z**2*a*b + za2b*z*a**2*b + zab2*z*a*b**2 + z2a2b*z**2*a**2*b + z2ab2*z**2*a*b**2 + za2b2*z*a**2*b**2 + z2a2b2*z**2*a**2*b**2 - kr

poor_opt3 = leastsq(res_ten3,  np.ones(26), args=(k_array, x_array, y_array,l_array))

print "Problem Solved (poor)"

rich_opt3 = leastsq(res_ten3,  np.ones(26), args=(k_array_rich, x_array, y_array,l_array))

print "Problem Solved (rich)"

## Unraveling parameters, creating k_star as a symbolic object ##

x1,x2,y1,y2,l1,l2,xy,xl,yl,x2y,xy2,x2y2,x2l,xl2,x2l2,y2l,yl2,y2l2,xyl,x2yl,xy2l,xyl2,x2y2l,x2yl2,xy2l2,x2y2l2 = poor_opt3[0]
z1,z2,a1,a2,b1,b2,za,zb,ab,z2a,za2,z2a2,z2b,zb2,z2b2,a2b,ab2,a2b2,zab,z2ab,za2b,zab2,z2a2b,z2ab2,za2b2,z2a2b2 = rich_opt3[0]

k, x, y, l, r = sym.var('k, x, y, l, r')

k_star_p = x1*x + x2*x**2 + y1*y + y2*y**2 + l1*l + l2*l**2 + xy*x*y + xl*x*l + yl*y*l + x2y*x**2*y + xy2*x*y**2 + x2y2*x**2*y**2 + x2l*x**2*l + xl2*x*l**2 + x2l2*x**2*l**2 + y2l*y**2*l + yl2*y*l**2 + y2l2*y**2*l**2 + xyl*x*y*l + x2yl*x**2*y*l + xy2l*x*y**2*l + xyl2*x*y*l**2 + x2y2l*x**2*y**2*l + x2yl2*x**2*y*l**2 + xy2l2*x*y**2*l**2 + x2y2l2*x**2*y**2*l**2
k_star_r = z1*x + z2*x**2 + a1*y + a2*y**2 + b1*l + b2*l**2 + za*x*y + zb*x*l + ab*y*l + z2a*x**2*y + za2*x*y**2 + z2a2*x**2*y**2 + z2b*x**2*l + zb2*x*l**2 + z2b2*x**2*l**2 + a2b*y**2*l + ab2*y*l**2 + a2b2*y**2*l**2 + zab*x*y*l + z2ab*x**2*y*l + za2b*x*y**2*l + zab2*x*y*l**2 + z2a2b*x**2*y**2*l + z2ab2*x**2*y*l**2 + za2b2*x*y**2*l**2 + z2a2b2*x**2*y**2*l**2

print k_star_p, k_star_r

## Saving results ##

results = open("k_params.txt", "w")
results.write("Poor:\n")
results.write(str(k_star_p)+"\n")
results.write("Rich:\n")
results.write(str(k_star_r)+"\n")
results.write("Parameters:\n")
results.write(str(poor_opt3[0])+"\n")
results.write(str(rich_opt3[0])+"\n")
results.close()