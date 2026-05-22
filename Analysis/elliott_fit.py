from datetime import datetime
from pathlib import Path
from optical_jomarie.absorption.elliott import elliott as ab
import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
import lmfit
from lmfit import Model
#constants
kb = 1.380649e-23 #(m^2kg/s^2/K)

def section(data, ref_col, use_col, lb, ub):
    nm = data[:,ref_col]
    idx = np.where((nm > lb) & (nm < ub))
    return(data[idx,use_col].transpose())
#S1 se data
data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/UV-Vis/Glass1_UV-Vis.csv', delimiter = ',')
eV = 1240/section(data, 0, 0, 387.5, 775) #energy
transmish = section(data,0,2,387.5,775) #smoothed intensity
#eV = 1240/data[:,0]
#transmish = data[:,2]
absorbs = -np.log10(transmish)

original_params = dict(eV=eV, a = 0.058, b = 1.79, n = 1, alpha = 2.75, Eg = 2.51, Ry = 0.089)

elliott_mod = ab.elliott(eV=eV, a = 0.058, b = 1.79, n = 1, alpha = 2.75, Eg = 2.5, Ry = 0.089)
#result1 = elliott_mod.fit(absorbs, eV=eV, a = 0.058, b = 1.79, n = 1, alpha = 2.75, Eg = 2.0, Ry = 0.089)
 #result2 = elliott_mod.fit(absorbs, eV=eV, a = 0.058, b = 1.79, n = 1, alpha = 2.75, Eg = 2.51, Ry = 0.089)
#print(result1.fit_report())

plt.plot(eV, absorbs, color = '#1f77b4', label = 'S1 Data')
plt.plot(eV, elliott_mod, '--', label = "Elliott Fit")
#plt.plot(eV, result1.best_fit, '-', label = "Best Fit")
plt.xlabel('Energy (eV)')
plt.ylabel('Absorbance')
plt.xlim(1.6,3.2)
plt.legend()
plt.show()

