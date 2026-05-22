import matplotlib.pyplot as plt
import numpy as np
import sympy
import scipy.constants as c
from sympy.parsing import sympy_parser
from optical_jomarie.general import organize as org
from optical_jomarie.general import ez_plots as ez
import lmfit
from scipy import interpolate

kb = c.k/c.e
data_PL1 = np.loadtxt('data/S1_SS_norm_smooth.csv', delimiter=',', comments='#') #comment die eerste ry met al die titles uit in die .csv of del dai ry
eV = ez.section(data_PL1, 0, 0, 1.6, 3.2)
PL = ez.section(data_PL1, 0, 1, 1.6, 3.2) #2de col
PL_norm = PL/np.max(PL)

rashba = sympy_parser.parse_expr('Er0 + A/(exp(Eph/kb*T)+1)')