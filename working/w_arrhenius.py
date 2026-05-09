import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
import lmfit

kb = c.k/c.e
data = np.loadtxt('data/S2_area_2-3_eV.txt', delimiter = ',', skiprows=1)
T = data[:,0]
I = data[:,1]

arr_1 = sympy_parser.parse_expr('Io/(1+A*exp(-Eb/(8.6173e-5*T)))')
gauss_1 = sympy_parser.parse_expr('sigma1*exp(-(T-T0)**2/(2*sigma2**2))')
expo_1 = sympy_parser.parse_expr('B*exp(-T/T_w + 1)')
model_list = sympy.Array((arr_1, gauss_1, expo_1))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

initial_param_values = dict(T=T, Io=23e4, A=10e4, Eb = 0.183, B=5e5, T_w = 80, sigma1 = 20e4, T0 = 243, sigma2 = 6)
np.random.seed(1)
param_values = dict(T=T, Io=23e4, A=10e4, Eb = 0.183, B=5e5, T_w = 80, sigma1 = 20e4, T0 = 243, sigma2 = 6)
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

#res.plot_fit()
#plt.plot(T, I, label='true')
#for c in yi:
#    plt.plot(T, c, color='0')
plt.plot(T, I)
plt.plot(T, res.best_fit)
plt.xlabel('Temperature (K)')
plt.ylabel('Integrated Intensity (a.u.) ')
plt.legend()
plt.show()