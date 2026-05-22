from datetime import datetime
from pathlib import Path
from optical_jomarie.absorption.elliott import elliott as ab
import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
import lmfit
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



