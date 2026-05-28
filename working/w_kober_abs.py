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
Eb = 0.229
sigma1 = 18
sigma2 = 61
sigmac = 100
Df = 4
b = 0.5 #gebasseer op my elliott code
A = 14.47

thresh = Eg-Eb
mask = eV > Eg
eV_mask = eV[mask]

sigma = np.where(eV <= thresh, sigma1, sigma2)

coeff = (Df*4*np.pi*Eb**(3/2))
## alpha1s = ((1/np.sqrt(2*np.pi*sigma**2))*np.exp(-0.5*(((eV-(Eg-Eb))/(sigma**2))**2))) #w/o broadening
#heaviside broadening werk ook nie - kyk of die twee convolutions in van dirac delta broadenings
alpha1s = ((1/np.sqrt(2*np.pi*sigma1))*np.exp(-0.5*(((-eV_mask+(Eg-Eb))/(sigma1**2))**2))) #with broadening - die arctan een is nie reg nie probeer die een van arednse
# die sigma threshold ding maak die transition te abrupt
alpha2s = ((1/np.sqrt(2*np.pi*sigma2))*np.exp(-0.5*(((eV_mask-(Eg-Eb))/(sigma2**2))**2)))
alphams = sum(((4*np.pi*(Eb**(3/2)))/(m**3))*(1/np.sqrt(2*np.pi*sigma1**2))*(np.exp(-0.5*(((eV-((Eg-Eb)/(m**2)))/(sigma1**2)))**2)) for m in range(2,12))
alphas = alpha1s+alpha2s
alpha_ex = (coeff*(alpha1s + alpha2s) + alphams)*((1/(b*np.sqrt(np.pi)))*np.exp(-(eV-(Eg-Eb))/b))

SommerF = ((2/np.sqrt(2*np.pi*sigmac**2)))*(np.exp(-0.5*(eV/(sigmac**2))**2))
alphac0 = np.where(eV>Eg, ((2*np.pi*eV)/(1-np.exp(-2*np.pi*eV)))*np.sqrt(eV-Eg),0)

alphac = np.where(eV > Eg, SommerF*alphac0, 0) #hoekom het ek die np.where

alpha = A*(alpha_ex + alphac)
print(np.shape(eV_mask))
plt.plot(eV, alpha1s, label='alpha1s')
plt.plot(eV, alpha2s, label='alpha2s')
plt.plot(eV, alphams, label='alphams')
plt.plot(eV, alpha_ex, label='alpha_ex')
plt.plot(eV, alphas, label='alphas')
plt.plot(eV, alphac, label='alphac')
plt.plot(eV, alpha, label='alpha')
plt.legend()
plt.show()
