import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('data/S1_centroids_Fitted.txt', delimiter=',',skiprows=1)
T = data[:,0]
energy = data[:,1]
fwhm = data[:,2]
height = data[:,3]/250

plt.scatter(T, energy, color='#1f77b4')
plt.xlabel('Temperature (K)')
plt.ylabel('PL Peak Energy (eV)')
plt.ylim(2.350,2.365)
plt.savefig('figures/S1_peak_energy.png')
plt.show()