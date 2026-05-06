import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
import lmfit

kb = c.k/c.e

arr_1 = sympy_parser.parse_expr('Io/(1+A*exp(-Eb/(kb*T)))')
expo_1 = sympy_parser.parse_expr('B*exp(-T/T_w + 1)')
model_list = sympy.Array((arr_1, expo_1))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

np.random.seed(1)
T = np.linspace(83, 293, 23)
param_values = dict(T=T, Io=23e4, A=10e4, kb = kb, Eb = 0.183, B=5e5, T_w = 80)
y = model_func(**param_values)
yi = model_list_func(**param_values)
yn = y + np.random.randn(y.size)*0.4

plt.plot(T, yn, 'o')
plt.plot(T, y)
for c in yi:
    plt.plot(T, c, color='0.7')

lm_mod = lmfit.Model(model_func, independent_vars=('T',))
res = lm_mod.fit(data=yn, **param_values)

res.plot_fit()
plt.plot(T, y, label='true')
plt.legend()

#model2 = model.subs('Io', 'T_w').subs('A', 'B')
model2_func = sympy.lambdify(list(model2.free_symbols), model2)
lm_mod = lmfit.Model(model2_func, independent_vars=('T',))
param2_values = dict(T=T, Io = 23e5, A=10e4,  kb = kb, Eb = 0.183, B=5e5, T_w=80)
res2 = lm_mod.fit(data=yn, **param2_values)
res2.plot_fit()
plt.plot(T, y, label='true')
plt.legend()
plt.show()