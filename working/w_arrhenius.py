import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
from optical_jomarie.general import organize as org
import lmfit
from scipy import interpolate

kb = c.k/c.e
data = np.loadtxt('data/S1_integrated_intensity.txt', delimiter = ',', skiprows=1)
T = data[:,0]
I = data[:,1]


temp = np.linspace(83,293,10000)
arr_1 = sympy_parser.parse_expr('Io/(1+A*exp(-Eb/(kb*T)))')
gauss_1 = sympy_parser.parse_expr('sigma1*exp(-(T-T0)**2/(2*sigma2**2))')
expo_1 = sympy_parser.parse_expr('B*exp(-T/T_w)-1')
gauss_2 = sympy_parser.parse_expr('sigma3*exp(-(T-T1)**2/(2*sigma4**2))')
model_list = sympy.Array((arr_1, gauss_1, expo_1, gauss_2))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

initial_param_values = dict(T=T, Io=23e4, A=10e4, Eb = 0.183, B=5e5, T_w = 80, sigma1 = 20e4, T0 = 243, sigma2 = 6)

param_values = lmfit.Parameters()
param_values.add('Io', value = 61)
param_values.add('A', value=196)
param_values.add('Eb', value = 0.120, vary = False)
param_values.add('kb', value = kb, vary=False)
param_values.add('sigma1', value=300, min=0, max = 600)
param_values.add('T0', value=243)
param_values.add('sigma2', value = 30)
param_values.add('B', value = 241)
param_values.add('T_w', value = 389)
param_values.add('sigma3', value=2.6, min=0, max = 600)
param_values.add('T1', value=98)
param_values.add('sigma4', value = 2.4)

lm_mod = lmfit.Model(model_func, independent_vars=('T',))
res = lm_mod.fit(data=I, T=T, params=param_values)

best_param_dict = {name: res.params[name].value for name in param_values}
yi_fitted = model_list_func(T=T, **best_param_dict)

print(res.fit_report())

best_vals = {sym: res.params[str(sym)].value
             for sym in model.free_symbols
             if str(sym) != 'T'}
fitted_eq = sympy.simplify(model.subs(best_vals))

print("Fitted equation:")
print(fitted_eq)

f = interpolate.interp1d(T, res.best_fit)
best = f(temp)
#res.plot_fit()
plt.scatter(T, I, color = '#ff7f0e', label='Data')
#for c in yi_fitted:
#    plt.plot(T, c, color='0.7')
plt.plot(temp, best, label='Fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Integrated Intensity (a.u.) ')
plt.legend()
plt.savefig('figures/s1_integ_int_fit.png')
plt.show()

save_or_nah = input('Do you want to save?').lower()

if save_or_nah in ('y', 'yes', 'slay'):
    out_dir = input('Where should it save to?')
    org.save_fit(f'fit_results/{out_dir}', x=T, y=I, x_name = 'Temperature (K)', y_name = 'Integrated Intensity', res = res, fitted_eq = fitted_eq, param_values=param_values)
elif save_or_nah in ('n', 'no', 'nah'):
    print('Okey-dokey')

