import numpy as np
import matplotlib.pyplot as plt
import optical_jomarie.general.ez_plots as ez
import optical_jomarie.general.peak_general as pg
import re 
import pandas as pd
import glob
from natsort import natsorted

path = "/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Glass1_S1/"
fns = glob.glob(path + '*.csv')

temp = []
head = []
for file in fns:
    pattern = r"\_S1/(.*?)\ "
    match = re.search(pattern, file)
    kelvin = int(match.group(1))+273
    temp.append(kelvin)
    head.append(str(kelvin) + ' K')

nm = np.loadtxt(fns[1], delimiter=',', usecols=0, skiprows=2)
eV = 1240/nm

all_dfs = pd.concat([pd.read_csv(one_filename, usecols= [2], skiprows = [0])
           for one_filename in fns], axis = 1).set_index(eV)
all_dfs.columns = temp #verander die naam van elke kol na sy respective temp toe sodat jy dit kan sort
all_dfs = all_dfs.reindex(natsorted(all_dfs.columns), axis=1) #gebruik a natural sorting algorithm
print(all_dfs)
df_sectioned = all_dfs[all_dfs.index.to_series().between(2,3)]
data = df_sectioned.to_numpy()
temp = natsorted(temp)
# synthetic data: Nx2 array
centroid1 = []
centroid2 = []


for i in range(23):
    y = data[:,i].transpose() #intensities of each temp in own col (23, 135)
    x = df_sectioned.index.to_series().to_numpy() 
    [min, max], fwhm, energy = pg.half_max_x(x, y)
    #min, fwhm, energy = pg.half_max_x(x, y)
    centroid2.append(energy)

print(energy)
plt.scatter(temp, centroid2, label = 'Centroid2')
plt.xlabel('Temperature (K)')
plt.ylabel('Energy (eV)')
plt.legend()
plt.show()