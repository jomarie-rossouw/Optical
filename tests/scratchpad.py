import scipy
from datetime import datetime
from pathlib import Path
#from optical_jomarie.absorption import elliott as ell
import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
import lmfit
import optical_jomarie.general as gen

kb = c.k/c.e
data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/PL/Glass1_S1_info_rem.csv', delimiter = ',')

nm = gen.ez_plots.section(data, 0, 0, 400, 650)
eV = 1240/nm
I = gen.ez_plots.section(data, 0, 1, 400, 650)

#bose_einstein = sympy_parser.parse_expr('Eg0-(2*a)/(exp(theta/T)-1)')
#fermi_dirac = sympy_parser.parse_expr('2/(exp(e - Eg0)/(8.617*T)+1)')
gauss_1 = sympy_parser.parse_expr('A*sigma1*exp(-(eV-E0)**2/(2*sigma1**2))')
gauss_2 = sympy_parser.parse_expr('D*sigma2*exp(-(eV-E03)**2/(2*sigma2**2))')
lorentz_1 = sympy_parser.parse_expr('B*gamma1/(2*pi*((eV-E01)**2)+((gamma1/2)**2))')
lorentz_2 = sympy_parser.parse_expr('C*gamma2/(2*pi*((eV-E02)**2)+((gamma2/2)**2))')
const = sympy_parser.parse_expr('c')
#expo_1 = sympy_parser.parse_expr('B*exp(-T/T_w + 1)')
model_list = sympy.Array((gauss_1, gauss_2, lorentz_1, lorentz_2))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

param_values = lmfit.Parameters()
param_values.add('A', value=1200)
param_values.add('D', value=1200)
param_values.add('B', value=400, min=0)
param_values.add('C', value=800, min=0)
#param_values.add('c', value=10, min=0, max=10)
param_values.add('sigma1', value=1, min=0, max=3)
param_values.add('sigma2', value=1, min=0, max=3)
param_values.add('E0', value=2.36, min=2, max=3)
param_values.add('E01', value=2.3, min=2, max=3)
param_values.add('E02', value=2.6, min=2, max=3)
param_values.add('E03', value=2.6, min=0, max=3)
#param_values.add('sigma2', value = 0.02, min=0, max=1)
param_values.add('gamma1', value=0.3, min=0, max=3)
#param_values.add('pi', value=np.pi, vary=False)
param_values.add('gamma2', value = 0.1, min=0, max=1)

#yi = model_list_func(E=eV,
               #sigma1=param_values['sigma1'].value,
               #E0=param_values['E0'].value,
               #gamma1=param_values['gamma1'].value,
               #gamma2=param_values['gamma2'].value)
               #pi=param_values['pi'].value)

lm_mod = lmfit.Model(model_func, independent_vars=('eV',))
res = lm_mod.fit(data=I, eV=eV, params=param_values)

best_param_dict = {name: res.params[name].value for name in param_values}
yi_fitted = model_list_func(eV=eV, **best_param_dict)

print(res.fit_report())

best_vals = {sym: res.params[str(sym)].value
                for sym in model.free_symbols
                if str(sym) != 'eV'}
fitted_eq = sympy.simplify(model.subs(best_vals))

print("Fitted equation:")
print(fitted_eq)

#fig = plt.figure()
#res.plot_fit()
res.plot_fit()
plt.plot(eV, yi_fitted[0], label='Fitted Gaussian 1', linestyle='--')
plt.plot(eV, yi_fitted[1], label='Fitted Lorentzian 1', linestyle='--')
plt.plot(eV, yi_fitted[2], label='Fitted Lorentzian 2', linestyle='--')
plt.plot(eV, yi_fitted[3], label='Fitted Gaussian 2', linestyle='--')
plt.plot(eV, I, label='Data')
plt.plot(eV, res.best_fit, label='Fitted')
plt.legend()
plt.show()
