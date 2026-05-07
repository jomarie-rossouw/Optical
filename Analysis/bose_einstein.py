import json
import os
import platform
import scipy
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
import lmfit

kb = c.k/c.e
data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/PL/ verwerkde_data_working/S1_centroids_Fitted.txt', delimiter = ',', skiprows=1)
T = data[:,0]
I = data[:,1]

bose_einstein = sympy_parser.parse_expr('Eg0-(2*a)/(exp(theta/T)-1)')
#fermi_dirac = sympy_parser.parse_expr('2/(exp(e - Eg0)/(8.617*T)+1)')
#gauss_1 = sympy_parser.parse_expr('sigma1*exp(-(T-T0)**2/(2*sigma2**2))')
#expo_1 = sympy_parser.parse_expr('B*exp(-T/T_w + 1)')
model_list = sympy.Array((bose_einstein))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

np.random.seed(1)
param_values = dict(T=T, a = 0.085, theta = 1395, Eg0 = 0.27) #a en hteta gebaseer op Arendse 2023
y = model_func(**param_values)
yi = model_list_func(**param_values)

lm_mod = lmfit.Model(model_func, independent_vars=('T',))
res = lm_mod.fit(data=I, **param_values)

print(res.fit_report())

best_vals = {sym: res.params[str(sym)].value
                for sym in model.free_symbols
                if str(sym) != 'T'}
fitted_eq = sympy.simplify(model.subs(best_vals))

print("Fitted equation:")
print(fitted_eq)

#fig = plt.figure()
#res.plot_fit()
plt.plot(T, yi)
plt.show()
