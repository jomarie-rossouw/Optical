import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import re
import matplotlib.pyplot as plt
from natsort import natsorted
from lmfit import Parameters, Minimizer, minimize, report_fit
from lmfit.models import LorentzianModel, GaussianModel, ConstantModel
from lmfit.model import save_modelresult, save_model, load_model, load_modelresult
from tabulate import tabulate
from pathlib import Path
# Load path
path = "/home/jo-marie/Documents/faafo/PL/TDPL/Glass2_S1/"
fns = glob.glob(path + '*.csv')

# Sort temps from filename
temp = []
head = []
for file in fns:
    pattern = r"\_S1/(.*?)\ "
    match = re.search(pattern, file)
    kelvin = int(match.group(1))+273
    temp.append(kelvin)
    head.append(str(kelvin) + ' K')

 # Load Data
nm = np.loadtxt(fns[1], delimiter=',', usecols=0, skiprows=2)
eV = 1240/nm

all_dfs = pd.concat([pd.read_csv(one_filename, usecols= [2], skiprows = [0])
           for one_filename in fns], axis = 1).set_index(eV)
all_dfs.columns = temp #verander die naam van elke kol na sy respective temp toe sodat jy dit kan sort
all_dfs = all_dfs.reindex(natsorted(all_dfs.columns), axis=1) #gebruik a natural sorting algorithm
all_dfs_norm = all_dfs/500

filepath = Path("data/S2_norm.csv")
filepath.parent.mkdir(parents=True, exist_ok=True)
all_dfs_norm.to_csv(filepath)
