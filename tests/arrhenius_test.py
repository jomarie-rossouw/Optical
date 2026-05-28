import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
import lmfit

kb = c.k/c.e
data = np.loadtxt('data/S1_integrated_intensity.txt', delimiter = ',', skiprows=1)
T = data[:,0]
I = data[:,1]

arr_1 = sympy_parser.parse_expr('Io/(1+A*exp(-Eb/(8.6173e-5*T)))')
gauss_1 = sympy_parser.parse_expr('(1/sigma1)*exp(-(T-T0)**2/(2*sigma1**2))')
expo_1 = sympy_parser.parse_expr('B*exp(-T/T_w + 1)')
model_list = sympy.Array((arr_1, gauss_1, expo_1))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

params = lmfit.Parameters()
params.add('Io', value=1000, min=0, max=10000)
params.add('A', value=60)
params.add('Eb', value=0.18, min=0.0005, max=0.8)
params.add('sigma1', value=40, min=40, max=100)
params.add('T0', value=243, min=200, max=263)
params.add('B', value=400)
params.add('T_w', value=30, min=0, max=350)
y = model_func(T=T, **params)
yi = model_list_func(T = T,**params)

lm_mod = lmfit.Model(model_func, independent_vars=('T',))
res = lm_mod.fit(data=I, T=T, **params)

print(res.fit_report())

best_vals = {sym: res.params[str(sym)].value
             for sym in model.free_symbols
             if str(sym) != 'T'}
fitted_eq = sympy.simplify(model.subs(best_vals))

print("Fitted equation:")
print(fitted_eq)

# res.plot_fit()
plt.plot(T, I, label='true')
plt.plot(T, res.init_fit)
#for c in yi:
#    plt.plot(T, c, color='0')

plt.legend()
plt.show()