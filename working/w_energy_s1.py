import json
import os
import platform
import scipy
import sys
from datetime import datetime
from pathlib import Path
from optical_jomarie.general import organize as org
import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
import lmfit

kb = c.k/c.e
data = np.loadtxt('data/S1_centroids_Fitted.txt', delimiter = ',', skiprows=1)
T = data[:,0]
eV = data[:,1]

T1 = T[0:12]
eV1 = eV[0:12]

T2 = T[11:18]
eV2 = eV[11:18]

T3 = T[17:23]
eV3 = eV[17:23]
bose_einstein = sympy_parser.parse_expr('Eg0-2*a/(exp(theta/T)-1)')
fermi_dirac = sympy_parser.parse_expr('Eg1 + A/(exp(theta1/T)+1)')
varshni = sympy_parser.parse_expr('Eg2 + (ab*T**2)/(T + wD)') #wD = hbar*omega_D/k_B (so die debye energy gedeel deur boltzmann - ons soek slegs die debye E so maaal dus die result met 8.617e-5)
#gauss_1 = sympy_parser.parse_expr('sigma1*exp(-(T-T0)**2/(2*sigma2**2))')
#expo_1 = sympy_parser.parse_expr('B*exp(-T/T_w + 1)')
model_list1 = sympy.Array((fermi_dirac))
model_list2 = sympy.Array((varshni, fermi_dirac))
model_list3 = sympy.Array((bose_einstein, varshni))
model1 = sum(model_list1)
model2 = sum(model_list2)
model3 = sum(model_list3)

model_list_func1 = sympy.lambdify(list(model_list1.free_symbols), model_list1)
model_func1 = sympy.lambdify(list(model1.free_symbols), model1)

param_values1 = lmfit.Parameters()
param_values1.add('A', value = 30)
param_values1.add('Eg1', value = 2.4, min=1, max=3)
param_values1.add('theta1', value = 300)
# param_values1.add('Eg2', value = 2, min=1, max=3)
# param_values1.add('wD', value=0.06)
# param_values1.add('ab', value=0.0008)
#param_values1.add('kb', value = 1, vary = False)

model_list_func2 = sympy.lambdify(list(model_list2.free_symbols), model_list2)
model_func2 = sympy.lambdify(list(model2.free_symbols), model2)

param_values2 = lmfit.Parameters()
param_values2.add('Eg2', value = 1.5, min=0, max = 4)
param_values2.add('wD', value=0.05)
param_values2.add('ab', value=0.0008)
param_values2.add('A', value = -30)
param_values2.add('Eg1', value = 2.36, min=1, max=3)
param_values2.add('theta1', value = -350)
#param_values2.add('kb', value = kb, vary = False)

model_list_func3 = sympy.lambdify(list(model_list3.free_symbols), model_list3)
model_func3 = sympy.lambdify(list(model3.free_symbols), model3)

param_values3 = lmfit.Parameters()
param_values3.add('Eg0', value = 2.356, min=1, max=3)
param_values3.add('a', value = 30)
param_values3.add('theta', value = 100)
param_values3.add('Eg2', value = 2.362, min=1, max=3)
param_values3.add('wD', value=10)
param_values3.add('ab', value=-0.01)

#param_values3.add('kb', value = kb, vary = False)


y1 = model_func1(T=T1, **param_values1)
y1i = model_list_func1(T=T1,**param_values1)

lm_mod1 = lmfit.Model(model_func1, independent_vars=('T',))
res1 = lm_mod1.fit(data=eV1, T=T1, params=param_values1)

best_param_dict1 = {name: res1.params[name].value for name in param_values1}
y1i_fitted = model_list_func1(T=T1, **best_param_dict1)

print(res1.fit_report())

best_vals1 = {sym: res1.params[str(sym)].value
                for sym in model1.free_symbols
                if str(sym) != 'T'}
fitted_eq1 = sympy.simplify(model1.subs(best_vals1))

y2 = model_func2(T=T2, **param_values2)
y2i = model_list_func2(T=T2,**param_values2)

lm_mod2 = lmfit.Model(model_func2, independent_vars=('T',))
res2 = lm_mod2.fit(data=eV2, T=T2, params=param_values2)

best_param_dict2 = {name: res2.params[name].value for name in param_values2}
y2i_fitted = model_list_func2(T=T2, **best_param_dict2)

print(res2.fit_report())

best_vals2 = {sym: res2.params[str(sym)].value
                for sym in model2.free_symbols
                if str(sym) != 'T'}
fitted_eq2 = sympy.simplify(model2.subs(best_vals2))

print("Fitted equation:")
print(fitted_eq2)

y3 = model_func3(T=T3, **param_values3)
y3i = model_list_func3(T=T3,**param_values3)

lm_mod3 = lmfit.Model(model_func3, independent_vars=('T',))
res3 = lm_mod3.fit(data=eV3, T=T3, params=param_values3)

best_param_dict3 = {name: res3.params[name].value for name in param_values3}
y3i_fitted = model_list_func3(T=T3, **best_param_dict3)

print(res3.fit_report())

best_vals3 = {sym: res3.params[str(sym)].value
                for sym in model3.free_symbols
                if str(sym) != 'T'}
fitted_eq3 = sympy.simplify(model3.subs(best_vals3))

print("Fitted equation:")
print(fitted_eq3)

#fig = plt.figure()
#res.plot_fit()
plt.scatter(T, eV, color = '#1f77b4')
plt.plot(T1, res1.best_fit, label = 'Model 1')
plt.plot(T2, res2.best_fit, label = 'Model 2')
plt.plot(T3, res3.best_fit, label = 'Model 3')
plt.xlabel('Temperature (K)')
plt.ylabel('Peak Energy (eV)')
# plt.savefig('figures/S1_energy_fit_v2.png')
plt.show()

# save_or_nah = input('Do you want to save?').lower()

# if save_or_nah in ('y', 'yes', 'slay'):
# out_dir = 'energy_s1'
# org.save_fit(f'fit_results/{out_dir}/model1_v2', x=T1, y=eV1, y_new=res1.best_fit, x_name = 'Temperature (K)', y_name = 'Peak Energy (eV)', res = res1, fitted_eq = fitted_eq1, model_list = model_list1, param_values=param_values1)
# org.save_fit(f'fit_results/{out_dir}/model2_v2', x=T2, y=eV2, y_new=res2.best_fit, x_name = 'Temperature (K)', y_name = 'Peak Energy (eV)', res = res2, fitted_eq = fitted_eq2, model_list = model_list2, param_values=param_values2)
# org.save_fit(f'fit_results/{out_dir}/model3_v2', x=T3, y=eV3, y_new=res3.best_fit, x_name = 'Temperature (K)', y_name = 'Peak Energy (eV)', res = res3, fitted_eq = fitted_eq3, model_list = model_list3, param_values=param_values3)
    
# elif save_or_nah in ('n', 'no', 'nah'):
    # print('Okey-dokey')
# 