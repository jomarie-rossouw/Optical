import numpy as np
from numpy import heaviside as Y
import matplotlib.pyplot as plt
from scipy import special
from scipy.special import erf as erf
import math 

'''gebasseer op die example: https://lmfit.github.io/lmfit-py/examples/example_sympy.html#sphx-glr-examples-example-sympy-py'''
''' vat as input die range waaroor jy die meetings geneem het, Ry, alpha, a, b, en n'''
def Exn(Eg, Ry, n): 
    '''discrete eigenenergies for boundstates'''
    return Eg-Ry/(n**2)

def Theta(eV, Eg, b):   #Heaviside Step Broadening
    return 1/(1+np.exp(-2*b*(eV-Eg))) 

def Gbf(sigma, z):
    return (1/sigma)*(np.exp((z)/sigma)/(1+np.exp((z)/sigma))**2)

def EBF_3D(A1, A2, E, Ry, n, Eg, disc, cont):
    summation=0
    for i in range(1, n+1):
        summation=(1/(n**3))*Gbf(disc, (Exn(Eg, Ry, n)-E))
    expo1 = 1/(1 + np.exp((Eg-E)/cont))
    expo2 = 1/(np.exp(2*np.pi*np.sqrt(Ry/(E-Eg)))-1)
    alpha_ebf = ((A1*np.sqrt(Ry))/E) * (2*Ry*summation + A2*(expo1 + expo2))
    return alpha_ebf

def alphaG_3D(A, E, Ry, n, Eg, sigma_0, cont):
    summ = 0
    disc = cont - (cont-sigma_0)/(n**2)
    for i in range(1 + n+1):
        summ = (np.exp(-(E - Eg + Ry/(n**2))**2/(2*disc**2))) / (n**3 * np.sqrt(2*np.pi) * disc)
    
    term2 = 0.5*(1 + erf((E-Eg)/(np.sqrt(2)*cont)))

    term3 = (cont/(58*np.sqrt(2*np.pi)))*np.exp(-(E-Eg-Ry)**2/(2*cont**2))

    term4 = ((E-Eg-Ry)/(116*Ry))*(1+erf((E-Eg-Ry)/(np.sqrt(2)*cont)))

    return ((A*np.sqrt(Ry))/E)*(2*Ry*summ + term2 + term3 + term4)

def alphaL_3D(A, E, Ry, n, Eg, Em, disc, cont):
    summ=0
    for i in range(1, n+1):
        summ = (disc/(n**3))/((E - Eg + Ry/(n**2))**2 + disc**2)
    term2 = 0.5 + np.arctan((E-Eg)/cont)/np.pi
    term3 = np.arctan((E-Eg-Ry)/cont)*((E-Eg-Ry)/(58*np.pi*Ry))
    term4 = -np.arctan((E - Em)/cont)*((E - Em)/(58*np.pi*Ry))
    term5 = (Em-Eg-Ry)/(116*Ry)
    term6 = -((cont/(116*np.pi*Ry))*np.log(((E-Eg-Ry)**2+cont**2)/((E-Em)**2+cont**2)))
    return (((A*np.sqrt(Ry))/E)*((2*Ry/np.pi)*summ + term2 + term3 + term4 + term5 + term6))

def alphaG_2D(A, sigma, E, Eg):
    return (A/(np.sqrt(2*np.pi)*sigma*E)) * np.exp(-(E-Eg)/(2*sigma**2))

def alphaL_2D(A, sigma, E, Eg, Ec): #moet positief wees
    term1 = sigma*(E-Eg)**2
    Theta = np.heaviside(E, Eg)     # die heaviside function Theta, H_step, en H_step sonder sigma lewer almal min of meer dieselfde result 
    H_step = (0.5 + np.arctan((E-Eg)/sigma)/np.pi) #toets eers met die sigma en dan sonder
    term3 = ((Ec**2-E**2)**2 + (sigma**2*E**2))
    return A*((term1*H_step)/term3)

def EBF_2D(A1, A2, E, Ry, n, Eg, sigma0, cont):
    summ = 0
    disc = cont - (cont-sigma0)/((n+1)**2)
    for i in range(1, int(n)+1):
        summ = 2*Ry * (Gbf(disc, (Eg - (Ry/((n+0.5)**2)) - E))) / (disc*(n+0.5)**3)
    term2 = 1/(1+np.exp((Eg-E)/cont))
    term3_1 = Theta(E, Eg, cont)/(np.exp(2*np.pi*np.sqrt(Ry/(E-Eg)))+1)
    term3 = -(0.5 + np.arctan((E-Eg))/np.pi)/(np.exp(2*np.pi*np.sqrt(Ry/(E-Eg)))+1) #vir E<Eg is dit obvi NaN, dis hoekom ons die heaviside step func het
    return (A1/E)*(summ + (term2+term3))

def EBF_2D_TL(A1, A2, E, Ry, n, Eg, B, Ec, disc, cont):
    summ = 0
    # disc = cont - (cont-sigma0)/((n+1)**2)
    for i in range(1, int(n)+1):
        summ = 2*Ry * (Gbf(disc, (Eg - (Ry/((n+0.5)**2)) - E))) / (disc*(n+0.5)**3)
    term2 = 1/(1+np.exp((Eg-E)/cont))
    term3_1 = Theta(E, Eg, cont)/(np.exp(2*np.pi*np.sqrt(Ry/(E-Eg)))+1)
    term3 = -(0.5 + np.arctan((E-Eg))/np.pi)/(np.exp(-2*np.pi*np.sqrt(Ry/(E-Eg)))+1) #vir E<Eg is dit obvi NaN, dis hoekom ons die heaviside step func het
    return (A1/E)*summ + (A2/E)*(term2+term3)*((B*E)/((Ec**2-E**2)**2+B**2*E**2))

if __name__ == '__main__':
    eV = np.linspace(1.5, 3.5, 1000)
    # A, Eg_TL, Ec_TL, sigma = 38, 2.426, 0.156, 0.214
    # A1, A2, Ry, n, Eg, disc, cont = 38, 36, 0.079, 1, 2.27, 0.0091, 0.214
    # A3, Ec_G, sigmaG = 30, 2.27, 0.0269
    # TL_2D = alphaL_2D(A, sigma, eV, Eg_TL, Ec_TL)
    # EBF_2d = EBF_2D(A1, A2, eV, Ry, n, Eg, disc, cont)
    # G_2D = alphaG_2D(A3, sigmaG, eV, Ec_G)
    
    A1, A2, Ry, n, Eg, disc, cont, B = 271e3, 8.1e3, 0.316, 1, 1.27, 0.02, 0.214, 365 #EBF_3D
    A, Eg_L, Em_L, gamma_n = 38, 2.426, 326, 0.05
    alpha =  (1)*alphaL_3D(A1, eV, Ry, n, Eg, Em_L, gamma_n, cont) + alphaG_3D(A1, eV, Ry, n, Eg, disc, cont) #+ (1)*EBF_3D(A1, A2, eV, Ry, n, Eg, disc, cont) 
    ebf = EBF_2D_TL(A1, A2, eV, Ry, n, Eg, B, Em_L, disc, cont)  
    TL = alphaL_2D(A, gamma_n, eV, Eg_L, Em_L)+ebf
    #cont = cont_spectra(alpha, gamma(Ry, eV))*heavi_broadening(eV, Eg, b)
    #print(Eb, E_n)
    #plt.plot(eV, Exo, '--k')
    # plt.xlim(2.15,2.45)
    print(alpha)
    plt.plot( eV, TL)
    plt.show()
     

