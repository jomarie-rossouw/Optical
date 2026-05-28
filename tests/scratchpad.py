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

data = np.loadtxt('data/S1_norm.csv', delimiter=',')
temp = data[0,1:]
eV = data[1:,0]
PL = data[1:,1:24] 
i = 22
T = temp[i]
print(T)
eV = ez.section(data, 0, 0, 1.6, 3.2)
PL = ez.section(data, 0, i, lb=1.6, ub=3.2) #2de col
pi = np.pi
print(np.max(PL))
print(eV[np.where(PL == np.max(PL))])

g1 = 0.17
sigma1 = 0.01
Eg_g1 = 2.362 #2.348, 2.35

l1 = 0.23
Gamma1 = 0.052
Eg_l1 = 2.36

g2 = 0.122
l2 = 0.01
Eg_g2 = 2.361
Eg_l2 = 2.637
sigma2 = 0.02
Gamma2 = 0.099

G1 = g1/((sigma1*np.sqrt(2*(np.pi))))*np.exp(-((eV-Eg_g1)**2)/(2*sigma1**2))
L1 = (l1/np.pi)*((0.5*Gamma1)/((eV-Eg_l1)**2+(0.5*Gamma1)**2))

plt.plot(eV, PL, label='Data')
# plt.plot(eV, G1, label='G')
plt.plot(eV, L1, label='L')
plt.title(f'{T} K')
plt.xlim(1.6,3.2)
plt.legend()
plt.show()