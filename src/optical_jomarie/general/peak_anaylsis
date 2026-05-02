import numpy as np 
import matplotlib.pyplot as plt #gaan likely nie gebruik nie

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(y - half)
    zero_crossings = np.where(signs[:-1] != signs[1:])[0]
    x_left = lin_interp(x, y, zero_crossings[1], half)
    x_right = lin_interp(x, y, zero_crossings[0], half)
    fwhm = x_right - x_left
    midpoint = 0.5*(x_right + x_left)
    print([x_left, x_right], fwhm, midpoint)

if name == '__main__':
    data = np.loadtxt('/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Glass1_S1/-190 -- 2026-Feb-16 12-17-08.csv', delimiter= ',')
    x = data[:,0]
    y = data[:,2]
    half_max_x(x, y)