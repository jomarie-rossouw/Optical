import matplotlib.pyplot as plt
import numpy as np

data = np.loadtxt('data/S1_area_2-3_eV.txt', delimiter=',',skiprows=1)
T = data[:,0]
integ = data[:,1]

plt.scatter(T, integ, color='#1f77b4')
plt.xlabel('Temperature (K)')
plt.ylabel('Integrated Intensity (a.u.)')
plt.ylim(0, 650)
plt.savefig('figures/S1_integ_int_2_3_eV.png')
plt.show()