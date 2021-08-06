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

fout = uproot3.recreate(f'results/{args.year}/{args.baseprocessor}/output.root')

for key, h in hists.items():
    if not isinstance(h, hist.Hist): continue
    for dataset in h.identifiers('dataset'):
        newhist = h.integrate('dataset', dataset)
        hname = '{}_{}'.format(dataset, key)
        #newhist.to_boost().to_numpy()
        fout[hname] = newhist.to_boost().to_numpy()

fout.close()