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
arr_1 = sympy_parser.parse_expr('Io1/(1+A*exp(-Eb1/(kb*T)))')
arr_2 = sympy_parser.parse_expr('Io3/(1+B*exp(-Eb3/(kb*T)))')
gauss_1 = sympy_parser.parse_expr('(Io2)*exp(-(kb*T-Eb2)**2/(2*sigma2**2))')
exp = sympy_parser.parse_expr('Io3*exp(-Eb3/kb*T)')
gauss_2 = sympy_parser.parse_expr('Io4*exp(-(kb*T-Eb4)**2/(2*sigma4**2))')
model_list = sympy.Array((arr_1, arr_2))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

# initial_param_values = dict(T=T, Io=23e4, A=10e4, Eb = 0.183, B=5e5, T_w = 80, sigma1 = 20e4, T0 = 243, sigma2 = 6)

param_values = lmfit.Parameters()
param_values.add('Io1', value = 9000, min=0, max=10000)
param_values.add('Io3', value = 220, min=0)
param_values.add('A', value=60)
param_values.add('Eb1', value = 0.116, min=0.01, max=0.250)
param_values.add('kb', value = kb, vary=False)
param_values.add('B', value=0.01) #assume it dies out before 0 K, met 0.1 is Bose Einstein vibes
param_values.add('Eb3', value=0.0206, min=0.01, max=0.300)

# param_values.add('Io1', value = 1, min=0, vary=False)
# param_values.add('A', value=196, vary=False)
# param_values.add('Eb1', value = 0.120, min=0, max=2, vary=False)
# param_values.add('kb', value = kb, vary=False)
# param_values.add('B', value=90000) #assume it dies out before 0 K, met 0.1 is Bose Einstein vibes
# param_values.add('Eb2', value=0.0206, vary=False) #hou by 0.0206
# param_values.add('sigma2', value =0.001) #hou by 0.001
# param_values.add('Io3', value = 473, min=0)
# param_values.add('Eb3', value = 0.0389, min=0)
# param_values.add('Io4', value=60, min=0)
# param_values.add('Eb4', value=0.014)
# param_values.add('sigma4', value = 0.010)

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
plt.scatter(T, I, color = '#1f77b4', label='Data')
#for c in yi_fitted:
#    plt.plot(T, c, color='0.7')
plt.plot(temp, best, label='Fit')
plt.xlabel('Temperature (K)')
plt.ylabel('Integrated Intensity (a.u.) ')
plt.legend()
# plt.savefig('figures/S1_integ_int_fit_v1.png')
plt.show()

# save_or_nah = input('Do you want to save?').lower()

# if save_or_nah in ('y', 'yes', 'slay'):
# out_dir = 'arr_s1'
# org.save_fit(f'fit_results/{out_dir}/integ_int_v1', x=T, y=I, y_new=res.best_fit, x_name = 'Temperature (K)', y_name = 'Integrated Intensity', res = res, fitted_eq = fitted_eq, model_list= model_list, param_values=param_values)
# elif save_or_nah in ('n', 'no', 'nah'):
#     # print('Okey-dokey')

