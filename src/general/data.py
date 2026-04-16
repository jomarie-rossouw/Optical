import glob
import numpy as np
import pandas as pd
from natsort import natsorted

def tdpl_import(path, pattern):
    #define directory of the files
    filename = glob.glob(path + '*.csv')

    #natsort the temps 
    temp = []
    header = []
    for file in filename:
        pattern =  r"\_S1/(.*?)\ "
