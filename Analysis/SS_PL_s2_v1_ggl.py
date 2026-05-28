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
data_PL1 = np.loadtxt('data/S2_SS_norm_smooth.csv', delimiter=',', comments='#') #comment die eerste ry met al die titles uit in die .csv of del dai ry
eV = ez.section(data_PL1, 0, 0, 1.6, 3.2)
PL = ez.section(data_PL1, 0, 1, 1.6, 3.2) #2de col
pi = np.pi
### 
G1 = sympy_parser.parse_expr('g1/((sigma1*sqrt(2*(psi))))*exp(-((eV-Eg_g1)**2)/(2*sigma1**2))')
G2 = sympy_parser.parse_expr('g2/((sigma2*sqrt(2*(psi))))*exp(-((eV-Eg_g2)**2)/(2*sigma2**2))')
L1 = sympy_parser.parse_expr('(l1/psi)*((0.5*Gamma1)/((eV-Eg_l1)**2+(0.5*Gamma1)**2))')
L2 = sympy_parser.parse_expr('(l2/psi)*((0.5*Gamma2)/((eV-Eg_l2)**2+(0.5*Gamma2)**2))')
# model_list = sympy.Array((G1, G2, L1, L2))
model_list = sympy.Array((G1, G2, L1))
model = sum(model_list)

model_list_func = sympy.lambdify(list(model_list.free_symbols), model_list)
model_func = sympy.lambdify(list(model.free_symbols), model)

'''G1 is for strong exciton-phonon coupling where the exciton density is assumed to be small such that the Probability P(Eb) is described by boltzman statistics '''
params = lmfit.Parameters()
params.add('psi', value=pi, vary=False)

g1 = 0.09
g2 = 0.122
l1 = 0.01
l2 = 0.01

Eg_g1 = 2.35 #2.348, 2.35
Eg_g2 = 2.361
Eg_l1 = 2.637
Eg_l2 = 2.637

sigma1 = 0.03
sigma2 = 0.02
Gamma1 = 0.1
Gamma2 = 0.099
params.add('g1', value=g1, min=0.001, max=1)
params.add('Eg_g1', value=Eg_g1, min=2.1, max=2.9)
params.add('sigma1', value = sigma1, min=0.01, max=0.1)

params.add('g2', value=g2, min=0.001, max=1)
params.add('Eg_g2', value=Eg_g2, min=2.1, max=2.9)
params.add('sigma2', value = sigma2, min=0.01, max=0.1)

params.add('l1', value=l1, min=0.001, max=1)
params.add('Eg_l1', value=Eg_l1, min=2.1, max=2.9)
params.add('Gamma1', value = Gamma1, min=0.01, max=0.1)

# params.add('l2', value=l2, min=0.001, max=1)
# params.add('Eg_l2', value=Eg_l2, min=2.1, max=2.9)
# params.add('Gamma2', value = Gamma2, min=0.01, max=0.1)

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

# labels = ['Gaussian 1', 'Gaussian 2', 'Lorentzian 1', 'Lorentzian 2']
labels = ['Gaussian 1', 'Gaussian 2', 'Lorentzian 1']
# labels = ['Gaussian 1', 'Gaussian 2']
# labels = ['Gaussian 1', 'Lorentzian 1', 'Lorentzian 2']
# labels = ['Lorentzian 1', 'Lorentzian 2']

# 
# for c, label in zip(yi_fitted, labels):
#    plt.plot(eV, c, label = label)
plt.plot(eV, PL, color = '#ff7f0e', label = 'Data')
plt.plot(eV, res.best_fit,'--', color="#042388", label = 'Model')
# plt.plot(eV, PL, label = 'Data')
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.xlim(1.6,3.2)
plt.ylim(ymin=None, ymax=3.5)
plt.savefig('figures/PL_f_SS_s2_.png')
plt.legend()
plt.show()

out_dir = 'SS_PL_Fit_S2'
# org.save_fit(f'fit_results/{out_dir}/v1', x=eV, y=PL, y_new=res.best_fit, x_name = 'Energy (eV)', y_name = 'Intensity (a.u.)', res = res, fitted_eq = fitted_eq, model_list = model_list, param_values=params)