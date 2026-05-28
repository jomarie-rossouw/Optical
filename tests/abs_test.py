import matplotlib.pyplot as plt
import numpy as np
from lmfit import CompositeModel, Model
from lmfit.lineshapes import gaussian, step
from datetime import datetime
from pathlib import Path
from optical_jomarie.absorption.elliott import elliott as ab
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
from optical_jomarie.absorption.elliott import EBF as ebf
from lmfit import Model
#constants
kb = 1.380649e-23 #(m^2kg/s^2/K)

def section(data, ref_col, use_col, lb, ub):
    nm = data[:,ref_col]
    idx = np.where((nm > lb) & (nm < ub))
    return(data[idx,use_col].transpose())

#S1 se data
data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/UV-Vis/Glass1_UV-Vis.csv', delimiter = ',')
eV = 1240/section(data, 0, 0, 400, 689) #energy
transmish = section(data,0,2,400,689) #smoothed intensity
absorbs = -np.log10(transmish)

def jump(x, mid):
    """Heaviside Step Function"""
    o = np.zeros(x.size)
    imid = max(np.where(x<=mid)[0])
    o[imid:] = 1.0
    return o

def convolve(arr, kernel):
    npts = min(arr.size, kernel.size)
    pad = np.ones(npts)
    tmp = np.concatenate((pad*arr[0], arr, pad*arr[-1]))
    out = np.convolve(tmp, kernel, mode='valid')
    noff = int((len(out)-npts)/2)
    return out[noff:noff+npts]

def alpha1s(eV, Eb, Eg, Df, sigma1, sigma2):
    coeff = (Df*4*np.pi*Eb**(3/2))
    sig1 = ((1/np.sqrt(2*np.pi*sigma1))*np.exp(-0.5*(((eV-(Eg-Eb))/(sigma1**2))**2))*(0.5+np.arctan((Eg-Eb)-eV)/np.pi))
    sig2 = ((1/np.sqrt(2*np.pi*sigma2))*np.exp(-0.5*(((eV-(Eg-Eb))/(sigma2**2))**2)))*(0.5+np.arctan(eV-(Eg-Eb))/np.pi)
    tot = coeff*(sig1+sig2)
    return sig1, sig2, tot

# def alphas(eV, Eb, Eg, Df, sigma1, sigma2):
#     coeff = (Df*4*np.pi*Eb**(3/2))
#     sigma = np.heaviside([sigma1, (Eg-Eb), sigma2], eV)
#     sig1 = ((1/np.sqrt(2*np.pi*sigma))*np.exp(-0.5*(((eV-(Eg-Eb))/(sigma**2))**2)))
#     tot = coeff*(sig1)
#     return sig1, tot

def alphams(eV, Eb, Eg, sigma1):
    alpha = 0
    for m in range(2,11):
        alpha = alpha + ((4*np.pi*(Eb**(3/2)))/(m**3))*(1/np.sqrt(2*np.pi*sigma1**2))*(np.exp(-0.5*(((eV-((Eg-Eb)/(m**2)))/(sigma1**2)))**2))
    return alpha

def alphac(eV, Eg, sigmac):
    b = 2*((1/(np.sqrt(2*np.pi*sigmac**2)))*np.exp(-0.5*(eV/(sigmac**2))**2))
    alphac0 = np.zeros(eV.size)
    for (eV > Eg):
        alphac0 = alphac0 + (((2*np.pi*eV)/(1-np.exp(-2*np.pi*eV)))*np.sqrt(eV-Eg))
    
    
s1, s2, tot1 = alpha1s(eV, 0.229, 2.63, 4.47, 18, 61)
tot = alphams(eV, 0.229, 2.63, 18)

fig, axes = plt.subplots(1,2,figsize=(12.8, 4.8))

# axes[0].plot(eV, tot1, 'bo')
axes[0].plot(eV, s1, 'k--', label='Sigma1')
axes[0].plot(eV, s2, 'r-', label='Sigma2')
axes[0].legend()

# axes[1].plot(eV, tot, 'bo')
axes[1].plot(eV, tot, 'k--', label='Sigma')
# axes[1].plot(eV, 10*comps['gaussian'], 'r-', label='Gaussian component')
axes[1].legend()
plt.show()

plt.show()