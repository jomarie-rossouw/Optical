import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
from optical_jomarie.general import organize as org
from optical_jomarie.general import ez_plots as ez
import lmfit
from lmfit.models import GaussianModel, LorentzianModel, DoniachModel
from scipy import interpolate

kbT = (c.k/c.e)*(300) #assume that measurements were taken at 300 K
data_PL1 = np.loadtxt('data/S1_SS_norm_smooth.csv', delimiter=',', comments='#') #comment die eerste ry met al die titles uit in die .csv of del dai ry
eV = ez.section(data_PL1, 0, 0, 1.6, 3.2)
PL = ez.section(data_PL1, 0, 1, 1.6, 3.2) #2de col
pi = np.pi
### 
G1 = sympy_parser.parse_expr('g1/((sigma1*sqrt(2*(psi))))*exp(-((eV-Eg_g1)**2)/(2*sigma1**2))')
G2 = sympy_parser.parse_expr('g2/((sigma2*sqrt(2*(psi))))*exp(-((eV-Eg_g2)**2)/(2*sigma2**2))')
L1 = sympy_parser.parse_expr('(l1/psi)*((0.5*Gamma1)/((eV-Eg_l1)**2+(0.5*Gamma1)**2))')
L2 = sympy_parser.parse_expr('(l2/psi)*((0.5*Gamma2)/((eV-Eg_l2)**2+(0.5*Gamma2)**2))')
model_list = sympy.Array((G1, G2, L1, L2))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

'''G1 is for strong exciton-phonon coupling where the exciton density is assumed to be small such that the Probability P(Eb) is described by boltzman statistics '''
params = lmfit.Parameters()
params.add('psi', value=pi, vary=False)

params.add('g1', value=0.1, min=0.001, max=1)
params.add('Eg_g1', value=2.6, min=2.1, max=2.9)
params.add('sigma1', value = 0.055, min=0.01, max=0.1)

params.add('g2', value=0.1, min=0.001, max=1)
params.add('Eg_g2', value=2.6, min=2.1, max=2.9)
params.add('sigma2', value = 0.055, min=0.01, max=0.1)

params.add('l1', value=0.1, min=0.001, max=1)
params.add('Eg_l1', value=2.6, min=2.1, max=2.9)
params.add('Gamma1', value = 0.055, min=0.01, max=0.1)

params.add('l2', value=0.1, min=0.001, max=1)
params.add('Eg_l2', value=2.6, min=2.1, max=2.9)
params.add('Gamma2', value = 0.055, min=0.01, max=0.1)

y = model_func(eV=eV, **params)
yi = model_list_func(eV=eV, **params)

mod = lmfit.Model(model_func, independent_vars=('eV',))
res = mod.fit(data=PL, eV=eV, params=params)

best_param_dict = {name: res.params[name].value for name in params}
yi_fitted = model_list_func(eV=eV, **best_param_dict)

print(res.fit_report())
best_vals = {sym: res.params[str(sym)].value
                for sym in model.free_symbols
                if str(sym) != 'eV'}
fitted_eq = sympy.simplify(model.subs(best_vals))

print(fitted_eq)

# for c in yi_fitted:
#    plt.plot(eV, c, color='0.7')
plt.plot(eV, PL, label = 'Data')
plt.plot(eV, res.best_fit,'--', color="#880404", label = 'Model')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.xlim(1.6,3.2)
plt.ylim(ymin=None, ymax=3.5)
plt.legend()
plt.savefig('figures/PL_f_SS_s1.png')
plt.show()

# out_dir = 'SS_PL_S2'
# org.save_fit(f'fit_results/{out_dir}/v0', x=eV, y=PL, y_new=res.best_fit, x_name = 'Energy (eV)', y_name = 'Intensity (a.u.)', res = res, fitted_eq = fitted_eq, model_list = model_list, param_values=params)