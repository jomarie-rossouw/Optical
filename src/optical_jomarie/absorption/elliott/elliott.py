import numpy as np
from numpy import heaviside as Y
import matplotlib.pyplot as plt
from scipy import special
from scipy.special import gamma as euler_L
import math 


''' vat as input die range waaroor jy die meetings geneem het, Ry, alpha, a, b, en n'''
def En(Ry, n, alpha): 
    '''discrete eigenenergies for boundstates'''
    return -Ry/(n + (alpha-3)/2)^2

def gamma(Ry, eV):
    return np.sqrt(Ry/eV)

def O0(alpha, eV, Ry):
    num = 2**(2*alpha-1) * np.array(eV) * (euler_L(alpha/2))**2 * euler_L((alpha-1)/2)
    denom = np.pi**((alpha-3)/2) * Ry * (euler_L(alpha-1))**3 
    return num/denom

def Exo_spectra(Ry, n, alpha):
    return (Ry * euler_L(n+alpha-2))/(math.factorial(n-1) * (n+(alpha-3)/2)**(alpha+1))

def dirac_broadening(Eg, En, eV, Ry, a, alpha, n):
    return 1/(a*np.sqrt(np.pi)) * np.exp( -(eV - En)/a)

def cont_spectra(alpha, gamma):
    num = np.abs(euler_L((alpha-1)/2  + j*gamma))**2 * np.exp(np.pi*gamma) * gamma**(2-alpha)
    denom = 2**alpha * np.pi**(2-alpha/2) * euler_L(alpha/2)
    return num/denom

def heavi_broadening(eV, b):
    return 1/(1+np.exp(-2*b*eV))