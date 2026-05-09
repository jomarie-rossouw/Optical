import numpy as np
import matplotlib.pyplot as plt
import optical_jomarie.general.ez_plots as ez
import optical_jomarie.general.peak_general as pg
import re 
import os
import pandas as pd
import glob
from natsort import natsorted
import sys
from pathlib import Path


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
        
#!/usr/bin/env python3
LINE_RE = re.compile(r"\['\s*([^']+)\s*'\]\s*\['\s*([^\]]+)\s*'\]")
FLOAT_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")

def parse_report(path):
    path = Path(path)
    text = path.read_text(encoding='utf-8', errors='ignore')
    params = {}
    for m in LINE_RE.finditer(text):
        name = m.group(1).strip()
        val = m.group(2).strip()
        fm = FLOAT_RE.search(val)
        if fm:
            params[name] = float(fm.group(0))
    return params

def to_latex(params):
    c = params.get('c', 0.0)
    parts = [f"{c:.8g}"]
    if all(k in params for k in ('l1_amplitude','l1_center','l1_sigma')):
        A = params['l1_amplitude']; x0 = params['l1_center']; g = params['l1_sigma']
        parts.append(r"\frac{{{A:.8g}}}{{\pi}}\cdot\frac{{{g:.8g}}}{{(x-{x0:.8g})^2+{g:.8g}^2}}".format(A=A,g=g,x0=x0))
    if all(k in params for k in ('l2_amplitude','l2_center','l2_sigma')):
        A = params['l2_amplitude']; x0 = params['l2_center']; g = params['l2_sigma']
        parts.append(r"\frac{{{A:.8g}}}{{\pi}}\cdot\frac{{{g:.8g}}}{{(x-{x0:.8g})^2+{g:.8g}^2}}".format(A=A,g=g,x0=x0))
    if all(k in params for k in ('g1_amplitude','g1_center','g1_sigma')):
        A = params['g1_amplitude']; mu = params['g1_center']; s = params['g1_sigma']
        parts.append(r"{A:.8g}\exp\!\left(-\frac{{(x-{mu:.8g})^2}}{{2{s:.8g}^2}}\right)".format(A=A,mu=mu,s=s))
    # join with + and wrap as a display equation
    body = " + ".join(parts)
    return r"\[ y(x) = " + body + r" \]"

def main(in_folder='.', out_folder='equations'):
    p = Path(in_folder)
    out_dir = Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in sorted(p.glob("S2_*_report.txt")):
        params = parse_report(f)
        tex = to_latex(params)
        out_name = f.stem.replace("_report", "_equation") + ".tex"
        out_path = out_dir / out_name
        out_path.write_text("% Auto-generated LaTeX equation\n" + tex + "\n", encoding='utf-8')
        print("Wrote", out_path)

if __name__ == '__main__':
    in_f = '/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Fitting_Done/S2_Fitted_Peaks/'
    out_f = 'S2/'
    params = parse_report(in_f)




#if __name__ == '__main__':
  #  path = "/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Glass1_S1/"
  #  outdir = '/home/jo-marie/Documents/optical_jomarie/analysis_results/S1'
  #  TDPL_concat(path, 2, 3, outdir)