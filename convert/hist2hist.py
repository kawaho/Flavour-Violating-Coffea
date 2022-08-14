from coffea.util import load
from statsmodels.stats.weightstats import DescrStatsW
from collections import defaultdict
import uproot
import numpy as np
import pandas as pd
import glob, os, json, argparse
import concurrent.futures
import coffea.hist
#Listing all systematics
metUnc = ['UnclusteredEn']
jetUnc = ['jesAbsolute', 'jesBBEC1', 'jesFlavorQCD', 'jesEC2', 'jesHF', 'jesRelativeBal']
jetyearUnc = sum([[f'jer_{year}', f'jesAbsolute_{year}', f'jesBBEC1_{year}', f'jesEC2_{year}', f'jesHF_{year}', f'jesRelativeSample_{year}'] for year in ['2017', '2018', '2016']], [])
sfUnc = sum([[f'pu_{year}', f'bTag_{year}'] for year in ['2017', '2018', '2016']], [])
sfUnc += ['pf_2016', 'pf_2017', 'mID', 'mIso', 'mTrg', 'eReco', 'eID']
theoUnc = [f'lhe{i}' for i in range(103)] + ['scalep5p5', 'scale22']
leptonUnc = ['ees', 'eer', 'me']

years = ['2016preVFP']#, '2016postVFP', '2017', '2018']

for masspt in [120,125,130]:  
  #Create dataframe for systematics: mva_sys/weight_sys/....etc 
  var_dict = {}
  hist_dict = {}
  cats = ['GG_GGcat', 'GG_VBFcat', 'VBF_GGcat', 'VBF_VBFcat']
  weight_year = {i:[] for i in cats}
  for year in years:
    print(f'Processing {year}')
    result = load(f"results/{year}/makeSys_reduce_{masspt}/output_herwig.coffea")
    if isinstance(result,tuple):
        result = result[0]
    for cat in cats:
      weight_year[cat].append(result[f'mva_hist-{cat}'].values()[()])
      print(masspt, year, cat, result[f'mva_hist-{cat}'].values()[()].sum())
    for varName in result:
      if 'hist' in varName:
        if varName in hist_dict:
          hist_dict[varName] = hist_dict[varName]+result[varName].values()[()]
        else:
          hist_dict[varName] = result[varName].values()[()]
      else:
        if varName in var_dict:
          var_dict[varName] = np.append(var_dict[varName], result[varName].value, axis=0)
        else:
          var_dict[varName] = result[varName].value
  df = pd.DataFrame(var_dict)
  #Separate into two dataframe: one for VBF cat and GG cat
  df_ggcat, df_vbfcat = df[(df['isVBFcat']==0)&(df['isHerwig']==0)], df[(df['isVBFcat']==1)&(df['isHerwig']==0)]
  
  #Create dataframe for fast calulcations of systematics
  for df_gg_vbf, whichcat in zip([df_ggcat, df_vbfcat], ['GGcat', 'VBFcat']):
    print(f'Running {whichcat}')
    #Get mva values corresponding to 100 quantiles 
    quantiles = np.load(f"results/SenScan/{whichcat}_quantiles",allow_pickle=True)
    #Divide into GG/VBF
    for df_gg_vbf_deep, whichcat_deep in zip([df_gg_vbf[df_gg_vbf['isVBF']==0], df_gg_vbf[df_gg_vbf['isVBF']==1]], ['GG', 'VBF']):
      print(f'Running {whichcat_deep} {masspt}')
      quan_dict = defaultdict(list)
  
      #Create tree for mva/weight/e_m_Mass/lep scale/smearing
      #mc
      datasets = []
      e_m_Mass = df_gg_vbf_deep[f'e_m_Mass'].to_numpy()
      weight = df_gg_vbf_deep['weight'].to_numpy()
      mva = df_gg_vbf_deep['mva'].to_numpy()

      h = coffea.hist.Hist("Observed bird count",
                     coffea.hist.Bin("e_m_Mass", "e_m_Mass", 7000, 100, 170),
                     coffea.hist.Bin("mva", "mva", 100, 0, 1),
                     )
      fillContent = {'e_m_Mass': e_m_Mass, 'mva': mva, 'weight': weight}
      h.fill(**fillContent)
      f = uproot.recreate(f"results/test2d.root")
      f['dict1'] = h.to_hist()

#      uproot_tree_dict_dtype = {'CMS_emu_Mass': e_m_Mass.dtype, 'mva': mva.dtype, 'weight': weight.dtype}
#      uproot_tree_dict = {'CMS_emu_Mass': e_m_Mass, 'mva': mva, 'weight': weight}
#      e_m_Mass_sys, mva_sys = [], []
#      for sys in leptonUnc:
#        for UpDown in ['Up', 'Down']:
#          e_m_Mass_sys.append(df_gg_vbf_deep[f'e_m_Mass_{sys}_{UpDown}'].to_numpy())
#          mva_sys.append(df_gg_vbf_deep[f'mva_{sys}_{UpDown}'].to_numpy())
#          uproot_tree_dict_dtype[f'CMS_emu_Mass_{sys}_{UpDown}'] = e_m_Mass_sys[-1].dtype
#          uproot_tree_dict[f'CMS_emu_Mass_{sys}_{UpDown}'] = e_m_Mass_sys[-1]
#          uproot_tree_dict_dtype[f'mva_{sys}_{UpDown}'] = mva_sys[-1].dtype
#          uproot_tree_dict[f'mva_{sys}_{UpDown}'] = mva_sys[-1]
#      with uproot3.recreate(f"results/SenScan/{whichcat_deep}_{whichcat}_{masspt}_tree.root") as f:
#        f['tree'] = uproot3.newtree(uproot_tree_dict_dtype)
#        f['tree'].extend(uproot_tree_dict)
