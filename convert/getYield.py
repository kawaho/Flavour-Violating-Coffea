from coffea.util import load
import coffea.hist
from statsmodels.stats.weightstats import DescrStatsW
from collections import defaultdict
import uproot
import numpy as np
import pandas as pd
import glob, os, json, argparse
import concurrent.futures

years = ['2016preVFP', '2016postVFP', '2017', '2018']
BRcorr = {120:{'GG':45.14/52.22, 'VBF':4.086/3.935}, 125:{'GG':41.98/48.61, 'VBF':3.925/3.766}, 130:{'GG':39.14/45.31, 'VBF':3.773/3.637}}

for masspt in [125, 110, 120, 130, 140, 150, 160]:  
  #Create dataframe for systematics: mva_sys/weight_sys/....etc 
  var_dict = {}
  hist_dict = {}
  cats = ['GG_GGcat', 'GG_VBFcat', 'VBF_GGcat', 'VBF_VBFcat']
  weight_year = {i:[] for i in cats}
  for year in years:
    print(f'Processing {year}')
    result = load(f"results/{year}/makeSys_reduce/output_v9_signal_{masspt}.coffea")
    if isinstance(result,tuple):
        result = result[0]
    for varName in result:
      if varName in ['weight', 'isVBF']:
        if varName in var_dict:
          var_dict[varName] = np.append(var_dict[varName], result[varName].value, axis=0)
        else:
          var_dict[varName] = result[varName].value
  
  #Get total weight of all years for acceptance calculations
  df = pd.DataFrame(var_dict)
  print(masspt,df.groupby('isVBF')['weight'].sum())
