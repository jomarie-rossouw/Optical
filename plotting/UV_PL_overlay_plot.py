import numpy as np
import matplotlib.pyplot as plt

def section(data, ref_col, use_col, lb, ub):
    nm = data[:,ref_col]
    idx = np.where((nm > lb) & (nm < ub))
    return(data[idx,use_col].transpose())

#extract data van csv
data = np.loadtxt('data_abs/Glass1_UV-Vis.csv', delimiter=',') #comment die eerste ry met al die titles uit in die .csv of del dai ry
nm = section(data, 0, 0, 350,775) #elke ry (:) van die eerste col (0)
eV = 1240/nm #convert nm na eV
UV_smooth = section(data, 0, 2, 350, 775)
absorbs = -np.log10(UV_smooth)
abs_norm = absorbs/np.max(absorbs)

data2 = np.loadtxt('data_abs/Glass2_UV-Vis.csv', delimiter=',') #comment die eerste ry met al die titles uit in die .csv of del dai ry
UV_smooth2 = section(data2, 0, 2, 350, 775)
absorbs2 = -np.log10(UV_smooth2)
abs_norm2 = absorbs2/np.max(absorbs2)

data_PL1 = np.loadtxt('data/S1_SS_norm_smooth.csv', delimiter=',', comments='#') #comment die eerste ry met al die titles uit in die .csv of del dai ry
PL1 = section(data_PL1, 0, 1, 1.6, 3.2) #2de col
PL_norm1 = PL1/np.max(PL1)

data_PL2 = np.loadtxt('data/S2_SS_norm_smooth.csv', delimiter=',', comments='#') #comment die eerste ry met al die titles uit in die .csv of del dai ry
nmPL2 = section(data_PL2, 0, 0, 1.6, 3.2) #elke ry (:) van die eerste col (0)
eVPL2 = 1240/nmPL2 #convert nm na eV
PL2 = section(data_PL2, 0, 1, 1.6, 3.2) #2de col
PL_norm2 = PL2/np.max(PL2)


# plt.plot(eV, PL1, color="#1f77b4", label='S1 Emission')
plt.plot(eV, abs_norm, color="#002174", label='Absorbance')
# plt.plot(eV, PL2, color="#ff7f0e", label='S2 Emission')
plt.plot(eV, abs_norm2, color="#924704ff", label='Absorbance')
plt.xlim(xmin=1.6)
# plt.ylim(ymax=3.5) #vir smoothed SS PL
#plt.title("(PEA)$_2$PbI$_4$ on Glass Sample 2 UV-Vis")
plt.xlabel("Energy (eV)")
plt.ylabel("Instensity (a.u)")
# plt.legend(loc = 'upper left')
# plt.savefig('figures/PL_u_overlay.png')
plt.show()

