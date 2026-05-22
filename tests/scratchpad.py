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

kb = c.k/c.e
data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/UV-Vis/Glass1_UV-Vis.csv', delimiter = ',')

nm = gen.ez_plots.section(data, 0, 0, 400, 650)
eV = 1240/nm
transmish = gen.ez_plots.section(data, 0, 1, 400, 650)
absorbs = -np.log10(transmish)
Eg = 2.6
Eb = 0.2
sigma1 = 18
sigma2 = 31
sigmac = 100
Df = 4
A = 14.47

thresh = Eg-Eb
mask = eV > Eg
eV_mask = eV[mask]

sigma = np.where(eV <= thresh, sigma1, sigma2)

coeff = (Df*4*np.pi*Eb**(3/2))
alpha1s = ((1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(((eV-(Eg-Eb))/(sigma**2))**2)))
alphams = sum(((4*np.pi*(Eb**(3/2)))/(m**3))*(1/np.sqrt(2*np.pi*sigma1**2))*(np.exp(-0.5*(((eV-((Eg-Eb)/(m**2)))/(sigma1**2)))**2)) for m in range(2,12))

alpha_ex = coeff*alpha1s + alphams

SommerF = ((2/np.sqrt(2*np.pi*sigmac**2)))*(np.exp(-0.5*(eV/(sigmac**2))**2))
alphac0 = np.where(eV>Eg, ((2*np.pi*eV_mask)/(1-np.exp(-2*np.pi*eV_mask)))*np.sqrt(eV_mask-Eg),0)

alphac = np.where(eV > Eg, SommerF*alphac0, 0)

alpha = A*(alphac)
 
plt.plot(eV, alpha, label='alpha')
plt.legend()
plt.show()
