#!/usr/bin/env python3
import sys
from pathlib import Path
import re
import csv

LINE_RE = re.compile(r"\['\s*([^']+)\s*'\]\s*\['\s*([^\]]+)\s*'\]")
FLOAT_RE = re.compile(r"[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?")

def parse_report(path):
    text = Path(path).read_text(encoding='utf-8', errors='ignore')
    params = {}
    for m in LINE_RE.finditer(text):
        name = m.group(1).strip()
        val = m.group(2).strip()
        fm = FLOAT_RE.search(val)
        if fm:
            params[name] = float(fm.group(0))
    return params

def main(in_folder='.', out_folder='.'):
    p = Path(in_folder)
    out_dir = Path(out_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p.glob("S*_report.txt"))
    if not files:
        print(p.glob("No files found.", file=sys.stderr))
        return
    
    all_params = []
    keys = set()
    for f in files:
        params = parse_report(f)
        name = f.stem.replace('_report','')
        params['file'] = name
        all_params.append(params)
        keys.update(params.keys())
    # ensure consistent ordering: file first
    header_keys = ['file'] + sorted(k for k in keys if k != 'file')
    out_path = out_dir / 'all_params.csv'
    with out_path.open('w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header_keys)
        for entry in all_params:
            row = [entry.get(k, "") for k in header_keys]
            writer.writerow(row)
    print(f"Wrote {out_path}")

if __name__ == '__main__':
    in_f = '/home/jo-marie/Documents/Experimental_11032026/PL/TDPL/Fitting_Done/S1_Fitted_Peaks'
    out_f = 'analysis_result/'
    main(in_f, out_f)
