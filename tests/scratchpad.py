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
data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/PL/ verwerkde_data_working/S1_area.txt', delimiter = ',', skiprows=1)
T = data[:,0]
I = data[:,1]

arr_1 = sympy_parser.parse_expr('Io/(1+A*exp(-Eb/(8.6173e-5*T)))')
gauss_1 = sympy_parser.parse_expr('sigma1*exp(-(T-T0)**2/(2*sigma2**2))')
expo_1 = sympy_parser.parse_expr('B*exp(-T/T_w + 1)')
model_list = sympy.Array((arr_1, gauss_1, expo_1))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

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

output_dir = Path('fit_results') / datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir.mkdir(parents=True, exist_ok=True)

np.savetxt(output_dir / 'data.csv',
            np.column_stack((T, I)),
            delimiter=',',
            header='T,I',
            comments='')

with open(output_dir / 'fit_report.txt', 'w') as f:
    f.write(res.fit_report())

with open(output_dir / 'fitted_equation.txt', 'w') as f:
    f.write(str(fitted_eq))

initial_params = {
    k: float(v) if np.isscalar(v) else np.asarray(v).tolist()
    for k, v in param_values.items()
    if k != 'T'
}
with open(output_dir / 'initial_parameters.json', 'w') as f:
    json.dump(initial_params, f, indent=2)

with open(output_dir / 'best_parameters.json', 'w') as f:
    json.dump(res.params.valuesdict(), f, indent=2)

repro_info = {
    'script': str(Path(__file__).resolve()),
    'date': datetime.now().isoformat(),
    'python_version': sys.version,
    'platform': platform.platform(),
    'numpy_version': np.__version__,
    'sympy_version': sympy.__version__,
    'lmfit_version': lmfit.__version__,
    'scipy_version': scipy.__version__,
}
with open(output_dir / 'reproduction_info.json', 'w') as f:
    json.dump(repro_info, f, indent=2)

fig = plt.figure()
res.plot_fit()
fig.savefig(output_dir / 'fit_plot.png', dpi=300)
plt.close(fig)

print(f"Saved fit results to {output_dir}")