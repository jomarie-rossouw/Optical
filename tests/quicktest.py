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
fns = "/home/jo-marie/Documents/Experimental_07052026/PL/Glass1_S1_info_rem.csv"
#fns = glob.glob(path + '*.csv')

# Sort temps from filename
 # Load Data
data = np.loadtxt(fns, delimiter=',')
eV = 1240/data[:,0]
I = data[:,1]/500
data = [[eV],[I]]
df = pd.DataFrame(data)
df_sectioned = all_dfs[all_dfs.index.to_series().between(2,3)]
data = df_sectioned.to_numpy()/500
temp = natsorted(temp)
filepath = Path("data/S2_norm.csv")
filepath.parent.mkdir(parents=True, exist_ok=True)
all_dfs_norm.to_csv(filepath)

