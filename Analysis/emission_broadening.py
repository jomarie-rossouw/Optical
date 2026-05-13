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
from optical_jomarie.general import organize as org

kb = c.k/c.e
data = np.loadtxt('data/S1_centroids_Fitted.txt', delimiter=',', skiprows=1)
T = data[:,0]
fwhm = data[:,2]

T1 = T
fwhm1 = fwhm

T2 = T[11:23]
fwhm2 = fwhm[11:23]

LA_and_0K = sympy_parser.parse_expr('Gamma_0 + LA*T')
LO_and_BE = sympy_parser.parse_expr('LO*(1/(exp(E_LO/(kb*T))-1))')
varshni = sympy_parser.parse_expr('((LO*T**2))/(T + wD)')
model_list1 = sympy.Array((LA_and_0K, LO_and_BE))
model_list2 = sympy.Array((LA_and_0K, varshni))
model1 = sum(model_list1)
model2 = sum(model_list2)

model_list_func1 = sympy.lambdify(list(model_list1.free_symbols), model_list1)
model_func1 = sympy.lambdify(list(model1.free_symbols), model1)

model_list_func2 = sympy.lambdify(list(model_list2.free_symbols), model_list2)
model_func2 = sympy.lambdify(list(model2.free_symbols), model2)

#die initial params gebruik ek van CsPbBr3 NCs net om n ball park te kry 
params1 = lmfit.Parameters()
params1.add('Gamma_0', value=0.05) #skuif dit op en af 5.9e-3 is goed
params1.add('LA', value=0.001) #23 is goed - skuif dit op en af
params1.add('LO', value=0.137,)
params1.add('kb', value=kb, vary=False)
params1.add('E_LO', value=0.030) #impacts the curve, 0.0x waardes is n reguit lyn, by 10 begin dit curve, 20-30 gee n decent fitting van 83-203 K, na 30 raak dit weer reguit

params2 = lmfit.Parameters()
params2.add('Gamma_0', value=20) #skuif dit op en af 5.9e-3 is goed
params2.add('LA', value=20) #23 is goed - skuif dit op en af
params2.add('LO', value=23) #impacts the curve, 0.0x waardes is n reguit lyn, by 10 begin dit curve, 20-30 gee n decent fitting van 83-203 K, na 30 raak dit weer reguit
params2.add('wD', value=30)

y1 = model_func1(T=T1, **params1)
y1i = model_list_func1(T=T1,**params1)

y2 = model_func2(T=T2, **params2)
y2i = model_list_func2(T=T2,**params2)

lm_mod1 = lmfit.Model(model_func1, independent_vars=('T',))
res1 = lm_mod1.fit(data=fwhm1, T=T1, params=params1)

lm_mod2 = lmfit.Model(model_func2, independent_vars=('T',))
res2 = lm_mod2.fit(data=fwhm2, T=T2, params=params2)

best_param_dict1 = {name: res1.params[name].value for name in params1}
y1i_fitted = model_list_func1(T=T1, **best_param_dict1)

best_param_dict2 = {name: res2.params[name].value for name in params2}
y2i_fitted = model_list_func2(T=T2, **best_param_dict2)

print(res1.fit_report())

best_vals1 = {sym: res1.params[str(sym)].value
                for sym in model1.free_symbols
                if str(sym) != 'T'}
fitted_eq1 = sympy.simplify(model1.subs(best_vals1))

# print(res2.fit_report())

# best_vals2 = {sym: res2.params[str(sym)].value
#                 for sym in model2.free_symbols
#                 if str(sym) != 'T'}
# fitted_eq2 = sympy.simplify(model2.subs(best_vals2))

plt.scatter(T, fwhm, color = '#1f77b4')
plt.plot(T1, res1.best_fit, label = 'Model 1')
# plt.plot(T2, res2.best_fit, label = 'Model 2')
plt.xlabel('Temperature (K)')
plt.ylabel('FWHM (eV)')
plt.savefig('figures/S1_fwhm_fit_v1.png')
plt.show()

out_dir = 'fwhm_s1'
org.save_fit(f'fit_results/{out_dir}/fwhm1_v1', x=T1, y=fwhm1, y_new=res1.best_fit, x_name = 'Temperature (K)', y_name = 'FWHM (eV)', res = res1, fitted_eq = fitted_eq1, model_list = model_list1, param_values=params1)