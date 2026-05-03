import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import BSpline, make_interp_spline
from lmfit import Minimizer, Parameters

#constants
kb = 1.380649e-23 #(m^2kg/s^2/K)

#S1 se data
data = np.loadtxt('/home/jo-marie/Documents/Thesis_Writeup/Results/verwerkde_data/S1_centroids_Fitted.txt', delimiter = ',')
T = data[:,0] #temp
E = data[:,1] #energy aka centroid
logT = np.log(T)
logE = np.log(E)

def arrhenius(pars, T, data=None):
    A, B = pars['A'], pars['B'] 
    model = A * np.exp(-B/(1.380649e-23*T))
    if data is None:
        return model
    return model - data

def darrhenius(pars, T, data=None):
    A, B = pars['A'], pars['B'] 
    v = np.exp(-B/(1.380649e-23*T))/T^2
    return np.array([v, -A/(B/(1.380649e-23))*T*v, np.ones(len(T))])

def arrhenius_log(T: np.ndarray, logA: float, B: float) -> np.ndarray:
    return logA - B/(1.380649e-23*T)

params = Parameters()
params.add('A', value=20)
params.add('B', value=20)

A, B = 1, 0.2

min1 = Minimizer(arrhenius, params, fcn_args=(T,), fcn_kws={'data': E})
out1 = min1.leastsq()
fit1 = arrhenius(out1.params, T)

#plt.plot(T, E, 'b-', label = 'S1 Energy')
plt.plot(T, fit1, 'r-')
plt.show()

