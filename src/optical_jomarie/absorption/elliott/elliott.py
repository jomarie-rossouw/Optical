import numpy as np
from numpy import heaviside as Y
import matplotlib.pyplot as plt
from scipy import special
from scipy.special import gamma as euler_L
import math 

'''gebasseer op die example: https://lmfit.github.io/lmfit-py/examples/example_sympy.html#sphx-glr-examples-example-sympy-py'''
''' vat as input die range waaroor jy die meetings geneem het, Ry, alpha, a, b, en n'''
def En(Eg, Ry, n, alpha): 
    '''discrete eigenenergies for boundstates'''
    return Eg -Ry/(n + (alpha-3)/2)**2

def gamma(Ry, eV):
    return np.sqrt(Ry/eV)

def O0(alpha, eV, Ry):
    num = 2**(2*alpha-1) * np.array(eV) * (euler_L(alpha/2))**2 * euler_L((alpha-1)/2)
    denom = np.pi**((alpha-3)/2) * Ry * (euler_L(alpha-1))**3 
    return num/denom

def Exo_spectra(Ry, n, alpha):
    return (Ry * euler_L(n+alpha-2))/(math.factorial(n-1) * (n+(alpha-3)/2)**(alpha+1)) 

def dirac_broadening(Ry, Eg, eV, n, a, alpha):
    return 1/(a*np.sqrt(np.pi)) * np.exp(-1 * np.power( ( eV-Eg-(Ry)/(np.power( n+(alpha-3)/2, 2 )) )/a, 2 ))

def cont_spectra(alpha, gamma):
    num = euler_L((alpha-1)/2  - 1j*gamma)*euler_L((alpha-1)/2  + 1j*gamma) * np.exp(np.pi*gamma) * gamma**(2-alpha) # verander |gamma|^2 na gamma(+i)*gamma(-i)
    denom = 2**alpha * np.pi**(2-(alpha/2)) * euler_L(alpha/2)
    return num/denom

def heavi_broadening(eV, Eg, b):
    return 1/(1+np.exp(-2*b*(eV-Eg)))

# estimates alpha, Ry, En
# constant n, a, en b
def elliott(eV, a, b, n, alpha, Eg, Ry):
    Exo = 0
    for i in range(1, int(n) +1):
        n = n+1 #begin by n=1 ipv n=0 want anders doen mens (-1)! which DNE
        E_n = En(Eg, Ry, i, alpha)
        Exo += Exo_spectra(Ry,i,alpha)*dirac_broadening(Ry, Eg, eV, n, a, alpha)

    O_0 = O0(alpha, eV, Ry)
    return (Exo + cont_spectra(alpha, gamma(Ry, eV))*heavi_broadening(eV, Eg, b))

if __name__ == '__main__':
    eV = np.linspace(1.5, 3.5, 1000)
    alpha, a, b, Eg, Ry, n = 2.75, 0.058, 1.79, 2.51, 0.089, 1
    absorp = elliott(eV, a, b, n, alpha, Eg, Ry)
    #cont = cont_spectra(alpha, gamma(Ry, eV))*heavi_broadening(eV, Eg, b)
    #print(Eb, E_n)
    #plt.plot(eV, Exo, '--k')
    print(absorp)
    plt.plot(eV, absorp)
    plt.show()
     

