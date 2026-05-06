import numpy as np
import matplotlib.pyplot as plt
import optical_jomarie.general.ez_plots as ez
import optical_jomarie.general.peak_general as pg
import re 
import os
import pandas as pd
import glob
from natsort import natsorted

def TDPL_concat(path, Emin, Emax):
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
    df_sectioned = all_dfs[all_dfs.index.to_series().between(Emin,Emax)]
    data = df_sectioned.to_numpy()
    temp = natsorted(temp)
    # synthetic data: Nx2 array
    return data
        
        


if __name__ == '__main__':
    path = "/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Glass1_S1/"
    outdir = '/home/jo-marie/Documents/optical_jomarie/analysis_results/S1'
    TDPL_concat(path, 2, 3, outdir)