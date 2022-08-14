from coffea.util import load
from statsmodels.stats.weightstats import DescrStatsW
from collections import defaultdict
import uproot3
import numpy as np
import pandas as pd
import glob, os, json, argparse
import concurrent.futures

years = ['2016preVFP', '2016postVFP', '2017', '2018']

#Create dataframe: mva/weight/e_m_Mass....etc 
df_year = []
for year in years:
  print(f'Processing {year}')
  result = load(f"results/{year}/makeSys/output_WJet2.coffea")
  if isinstance(result,tuple):
      result = result[0]
  for varName in result:
    result[varName] = result[varName].value

  df_year.append(pd.DataFrame(result))
  df_year[-1] = df_year[-1][(df_year[-1].e_m_Mass>110) & (df_year[-1].e_m_Mass<160)]
  print(f'Finish {year}')

theory_total_weight = {}
df = pd.concat(df_year)

#Separate into two dataframe: one for VBF cat and GG cat
df_ggcat, df_vbfcat = df[(df['isVBFcat']==0)&(df['isHerwig']==0)], df[(df['isVBFcat']==1)&(df['isHerwig']==0)]

#Repeat for data
var_dict_data = {}
for year in years:
  print(f'Processing {year}')
  result = load(f"results/{year}/makeData/output.coffea")
  if isinstance(result,tuple):
      result = result[0]
  for varName in result:
    if varName in var_dict_data:
      var_dict_data[varName] = np.append(var_dict_data[varName],result[varName].value, axis=0)
    else:
      var_dict_data[varName] = result[varName].value
  df_data = pd.DataFrame(var_dict_data)
  df_data = df_data[df_data['weight']!=0]
df_ggcat_data, df_vbfcat_data = df_data[(df_data['isVBFcat']==0)], df_data[(df_data['isVBFcat']==1)]

#Create dataframe 
for df_gg_vbf, df_gg_vbf_data, whichcat in zip([df_ggcat, df_vbfcat], [df_ggcat_data, df_vbfcat_data], ['GGcat', 'VBFcat']):
  datasets = []
  e_m_Mass = df_gg_vbf_data[f'e_m_Mass'].to_numpy()
  mva = df_gg_vbf_data['mva'].to_numpy()
  with uproot3.recreate(f"results/SenScan/{whichcat}_tree_nosys.root") as f:
    f['tree'] = uproot3.newtree({'CMS_emu_Mass': e_m_Mass.dtype, 'mva': mva.dtype})
    f['tree'].extend({'CMS_emu_Mass': e_m_Mass, 'mva': mva})

  #Divide into GG/VBF
  for df_gg_vbf_deep, whichcat_deep in zip([df_gg_vbf[df_gg_vbf['isVBF']==0], df_gg_vbf[df_gg_vbf['isVBF']==1]], ['GG', 'VBF']):

    quan_dict = defaultdict(list)

    #Create tree for mva/weight/e_m_Mass
    #mc
    datasets = []
    e_m_Mass = df_gg_vbf_deep[f'e_m_Mass'].to_numpy()
    weight = df_gg_vbf_deep['weight'].to_numpy()
    mva = df_gg_vbf_deep['mva'].to_numpy()
    uproot_tree_dict_dtype = {'CMS_emu_Mass': e_m_Mass.dtype, 'mva': mva.dtype, 'weight': weight.dtype}
    uproot_tree_dict = {'CMS_emu_Mass': e_m_Mass, 'mva': mva, 'weight': weight}
    e_m_Mass_sys, mva_sys = [], []
    with uproot3.recreate(f"results/SenScan/{whichcat_deep}_{whichcat}_tree_nosys.root") as f:
      f['tree'] = uproot3.newtree(uproot_tree_dict_dtype)
      f['tree'].extend(uproot_tree_dict)
