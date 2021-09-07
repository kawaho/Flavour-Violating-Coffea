from coffea.util import load
from coffea import hist
import pandas as pd
import glob, os, json, argparse, uproot3

parser = argparse.ArgumentParser(description='Convert coffea output to THist')
parser.add_argument('-b', '--baseprocessor', type=str, default='makeHist', help='processor tag (default: %(default))')
parser.add_argument('-y', '--year', type=str, default=None, help='analysis year')
args = parser.parse_args()

hists = load(f"results/{args.year}/{args.baseprocessor}/output.coffea")
if isinstance(hists, tuple):
    hists = hists[0]

fout = {}

for key, h in hists.items():
    if not isinstance(h, hist.Hist): continue
    for dataset in h.identifiers('dataset'):
        if not dataset in fout:
           fout[dataset] = uproot3.recreate(f'results/{args.year}/{args.baseprocessor}/{dataset}.root')

for key, h in hists.items():
    if not isinstance(h, hist.Hist): continue
    for dataset in h.identifiers('dataset'):
        newhist = h.integrate('dataset', dataset)
        hname = f'{key}'
        fout[dataset][hname] = hist.export1d(newhist)

for f in fout:
    fout[f].close()

