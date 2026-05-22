import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import re
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import cm 
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from natsort import natsorted

data = np.loadtxt('data/S1_norm.csv', delimiter=',')
temp = data[0,18:25]
eV = data[1:,0]
I = data[1:,18:26]
colors = plt.cm.rainbow(np.linspace(1,0.2,6))

plt.figure(figsize=(10, 6))
for i in range(6):
    plt.plot(eV, I[:,i], color=colors[i], label=temp[i])
plt.xlim(2.4,3)
plt.xlabel('Energy (eV)')
plt.ylabel('Intensity (a.u.)')
plt.ylim(ymax=2)
plt.legend()
# plt.savefig('figures/PL_u_TDPL_overlay_s1_anom.png')
plt.show()
