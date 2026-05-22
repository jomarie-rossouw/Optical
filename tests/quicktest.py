import matplotlib.pyplot as plt
import numpy as np
from lmfit import CompositeModel, Model
from lmfit.lineshapes import gaussian, step
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
import optical_jomarie.general as gen

data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/PL/Glass1_S1_info_rem.csv', delimiter = ',')

nm = gen.ez_plots.section(data, 0, 0, 400, 650)
eV = 1240/nm
I = gen.ez_plots.section(data, 0, 1, 400, 650)

print((1/(0.5*np.sqrt(np.pi)))*np.exp(-((eV-2.6)/0.5)))
def jump(x, mid):
    """Heaviside Step Function"""
    o = np.zeros(x.size)
    imid = max(np.where(x<=mid)[0])
    o[imid:] = o*(1/(0.5*np.sqrt(np.pi)))*np.exp(-((x[imid:]-2.6)/0.5))
    return o

def convolve(arr, kernel):
    npts = min(arr.size, kernel.size)
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out)-npts)/2)
    return out[noff:noff+npts]

mod = CompositeModel(Model(jump), Model(gaussian), convolve)

pars = mod.make_params(amplitude=dict(value=1, min=0), center=1.5, sigma=dict(value=0.5, min=0), mid=dict(value=100, vary=False))

result = mod.fit(I, params=pars, x=eV)

print(result.fit_report())

comps = result.eval_components(x=eV)

fig, axes = plt.subplots(1,2,figsize=(12.8, 4.8))


axes[0].plot(eV, I, 'bo')
axes[0].plot(eV, result.init_fit, 'k--', label='initial fit')
axes[0].plot(eV, result.best_fit, 'r-', label='best fit')
axes[0].legend()

axes[1].plot(eV, I, 'bo')
axes[1].plot(eV, 10*comps['jump'], 'k--', label='Jump component')
axes[1].plot(eV, 10*comps['gaussian'], 'r-', label='Gaussian component')
axes[1].legend()
plt.show()

