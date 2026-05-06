import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import BSpline, make_interp_spline
import scipy.constants as c
from lmfit import Minimizer, Parameters
import optical_jomarie.PL.PL as PL

kb = c.k/c.e

def arr_model(I0, T, A, Eb):
    I_T = 














if __name__ == '__main__':
    data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Fitting_Done/S1_Area/S1_area.txt', delimiter=',')
