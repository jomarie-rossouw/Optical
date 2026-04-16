import glob
import re
import numpy as np
import pandas as pd
from natsort import natsorted

def myfunc(text, flag=re.NOFLAG):
    return re.match(text, flag)

re_test = myfunc(r'\_S1/\ ')