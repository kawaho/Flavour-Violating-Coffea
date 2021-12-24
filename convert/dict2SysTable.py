from coffea.util import load
from statsmodels.stats.weightstats import DescrStatsW
from collections import defaultdict
import uproot3
import numpy as np
import pandas as pd
import glob, os, json, argparse
import concurrent.futures

#Listing all systematics
metUnc = ['UnclusteredEn']
jetUnc = ['jesAbsolute', 'jesBBEC1', 'jesFlavorQCD', 'jesEC2', 'jesHF', 'jesRelativeBal']
jetyearUnc = sum([[f'jer_{year}', f'jesAbsolute_{year}', f'jesBBEC1_{year}', f'jesEC2_{year}', f'jesHF_{year}', f'jesRelativeSample_{year}'] for year in ['2017', '2018', '2016']], [])
sfUnc = sum([[f'pu_{year}', f'bTag_{year}'] for year in ['2017', '2018', '2016preVFP', '2016postVFP']], [])
sfUnc += ['pf_2016preVFP', 'pf_2016postVFP', 'pf_2017']
theoUnc = [f'lhe{i}' for i in range(103)] + ['scalep5p5', 'scale22']
leptonUnc = ['me']#['ees', 'eer', 'me']

years = ['2016preVFP', '2016postVFP', '2017', '2018']

#Create dataframe for systematics: mva_sys/weight_sys/....etc 
var_dict = {}
for year in years:
  print(f'Processing {year}')
  result = load(f"results/{year}/makeSys/output_test.coffea")
  if isinstance(result,tuple):
      result = result[0]
  for varName in result:
    if varName in var_dict:
      var_dict[varName] = np.append(var_dict[varName], result[varName].value, axis=0)
    else:
      var_dict[varName] = result[varName].value

#Get total weight of all years for acceptance calculations
theory_total_weight = {}
df = pd.DataFrame(var_dict)

theory_total_weight["weight_GG"] = df[(df['isVBF']==0)]["weight"].sum()
theory_total_weight["weight_scalep5p5_GG"] = df[(df['isVBF']==0)]["weight_scalep5p5"].sum()
theory_total_weight["weight_scale22_GG"] = df[(df['isVBF']==0)]["weight_scale22"].sum()
for i in range(103):
  theory_total_weight[f"weight_lhe{i}_GG"] = df[(df['isVBF']==0)][f"weight_lhe{i}"].sum()

theory_total_weight["weight_VBF"] = df[(df['isVBF']==1)]["weight"].sum()
theory_total_weight["weight_scalep5p5_VBF"] = df[(df['isVBF']==1)]["weight_scalep5p5"].sum()
theory_total_weight["weight_scale22_VBF"] = df[(df['isVBF']==1)]["weight_scale22"].sum()
for i in range(103):
  theory_total_weight[f"weight_lhe{i}_VBF"] = df[(df['isVBF']==1)][f"weight_lhe{i}"].sum()

#Separate into two dataframe: one for VBF cat and GG cat
df_ggcat, df_vbfcat = df[(df['isVBFcat']==0)&(df['isHerwig']==0)], df[(df['isVBFcat']==1)&(df['isHerwig']==0)]
#df_ggcat_herwig, df_vbfcat_herwig = df[(df['isVBFcat']==0)&(df['isHerwig']==1)], df[(df['isVBFcat']==1)&(df['isHerwig']==1)]

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

#Create dataframe for fast calulcations of systematics
for df_gg_vbf, df_gg_vbf_data, whichcat in zip([df_ggcat, df_vbfcat], [df_ggcat_data, df_vbfcat_data], ['GGcat', 'VBFcat']):
  #Get mva values corresponding to 100 quantiles 
  wq = DescrStatsW(data=df_gg_vbf['mva'], weights=df_gg_vbf['weight'])
  quantiles = wq.quantile(probs=np.linspace(0,1,101), return_pandas=False)
  quantiles[0], quantiles[-1] = 0, 1
  quantiles.dump(f"results/{whichcat}_quantiles")
  #with open(f"{whichcat}_quantiles.json", 'w') as f:
  #  json.dump(quantiles, f, indent=2) 
  #Create tree for mva/weight/e_m_Mass
  datasets = []
  e_m_Mass = df_gg_vbf_data[f'e_m_Mass'].to_numpy()
  mva = df_gg_vbf_data['mva'].to_numpy()
  with uproot3.recreate(f"results/{whichcat}_tree.root") as f:
    f['tree'] = uproot3.newtree({'CMS_emu_Mass': e_m_Mass.dtype, 'mva': mva.dtype})
    f['tree'].extend({'CMS_emu_Mass': e_m_Mass, 'mva': mva})

  #Divide into GG/VBF
  for df_gg_vbf_deep, whichcat_deep in zip([df_gg_vbf[df_gg_vbf['isVBF']==0], df_gg_vbf[df_gg_vbf['isVBF']==1]], ['GG', 'VBF']):

    quan_dict = defaultdict(list)

    #Create tree for mva/weight/e_m_Mass/lep scale/smearing
    #mc
    datasets = []
    e_m_Mass = df_gg_vbf_deep[f'e_m_Mass'].to_numpy()
    weight = df_gg_vbf_deep['weight'].to_numpy()
    mva = df_gg_vbf_deep['mva'].to_numpy()
    uproot_tree_dict_dtype = {'CMS_emu_Mass': e_m_Mass.dtype, 'mva': mva.dtype, 'weight': weight.dtype}
    uproot_tree_dict = {'CMS_emu_Mass': e_m_Mass, 'mva': mva, 'weight': weight}
    e_m_Mass_sys, mva_sys = [], []
    for sys in leptonUnc:
      for UpDown in ['Up', 'Down']:
        e_m_Mass_sys.append(df_gg_vbf_deep[f'e_m_Mass_{sys}_{UpDown}'].to_numpy())
        mva_sys.append(df_gg_vbf_deep[f'mva_{sys}_{UpDown}'].to_numpy())
        uproot_tree_dict_dtype[f'CMS_emu_Mass_{sys}_{UpDown}'] = e_m_Mass_sys[-1].dtype
        uproot_tree_dict[f'CMS_emu_Mass_{sys}_{UpDown}'] = e_m_Mass_sys[-1]
        uproot_tree_dict_dtype[f'mva_{sys}_{UpDown}'] = mva_sys[-1].dtype
        uproot_tree_dict[f'mva_{sys}_{UpDown}'] = mva_sys[-1]
    with uproot3.recreate(f"results/{whichcat_deep}_{whichcat}_tree.root") as f:
      f['tree'] = uproot3.newtree(uproot_tree_dict_dtype)
      f['tree'].extend(uproot_tree_dict)

    #Loop through quantiles and turn tree into RooDataSet
    def FillSys(i):
      print(f'Filling Sys for {whichcat_deep} {whichcat} Quantiles {i}')
      dict_ = []
      subdf = df_gg_vbf_deep[(df_gg_vbf_deep['mva']<quantiles[i+1])&(df_gg_vbf_deep['mva']>=quantiles[i])]
      #Store all the sys weight/acceptance
      dict_.append(['quantiles', i+1])
      dict_.append(['lowerMVA', quantiles[i]])
      dict_.append(['weight', subdf['weight'].sum()])
      dict_.append(["weight2016", subdf[subdf["is2016"]==1]["weight"].sum()])
      dict_.append(["weight2017", subdf[subdf["is2017"]==1]["weight"].sum()])
      dict_.append(["weight2018", subdf[subdf["is2018"]==1]["weight"].sum()])
      dict_.append(['acc', subdf['weight'].sum()/theory_total_weight[f'weight_{whichcat_deep}']])
#      if whichcat_deep=='VBF':
#        if whichcat=='VBFcat':
#          df_herwig = df_vbfcat_herwig
#        else:
#          df_herwig = df_ggcat_herwig
#        subdf = df_herwig[(df_herwig[f'mva']<quantiles[i+1])&(df_herwig[f'mva']>=quantiles[i])]
#        dict_.append([f'weight_herwig', subdf[f'weight'].sum()])
      for sys in theoUnc:
        dict_.append([f'weight_{sys}', subdf[f'weight_{sys}'].sum()/theory_total_weight[f'weight_{sys}_{whichcat_deep}']])
      for UpDown in ['Up', 'Down']:
        for sys in sfUnc:
          dict_.append([f'weight_{sys}_{UpDown}', subdf[f'weight_{sys}_{UpDown}'].sum()])
        for sys in leptonUnc+metUnc:
          subdf = df_gg_vbf_deep[(df_gg_vbf_deep[f'mva_{sys}_{UpDown}']<quantiles[i+1])&(df_gg_vbf_deep[f'mva_{sys}_{UpDown}']>=quantiles[i])]
          dict_.append([f'weight_{sys}_{UpDown}', subdf[f'weight'].sum()])
    
        for sys in jetUnc+jetyearUnc:
          if whichcat=='GGcat':
            if whichcat_deep=='GG':
              df_gg_vbf_deep_sys = df[(df[f'isVBFcat_{sys}_{UpDown}']==0) & (df[f'isVBF']==0)] 
            else:
              df_gg_vbf_deep_sys = df[(df[f'isVBFcat_{sys}_{UpDown}']==0) & (df[f'isVBF']==1)]
          else: 
            if whichcat_deep=='GG':
              df_gg_vbf_deep_sys = df[(df[f'isVBFcat_{sys}_{UpDown}']==1) & (df[f'isVBF']==0)] 
            else:
              df_gg_vbf_deep_sys = df[(df[f'isVBFcat_{sys}_{UpDown}']==1) & (df[f'isVBF']==1)]
    
          subdf = df_gg_vbf_deep_sys[(df_gg_vbf_deep_sys[f'mva_{sys}_{UpDown}']<quantiles[i+1])&(df_gg_vbf_deep_sys[f'mva_{sys}_{UpDown}']>=quantiles[i])]
          dict_.append([f'weight_{sys}_{UpDown}', subdf[f'weight'].sum()])
      return dict_


    #for i in range(len(quantiles)-1):
    #  FillSys(i)
    executor = concurrent.futures.ProcessPoolExecutor(32)
    futures = [executor.submit(FillSys, i) 
           for i in range(len(quantiles)-1)]
    long_dict_ = []
    for future in concurrent.futures.as_completed(futures):
      long_dict_.extend(future.result())
    concurrent.futures.wait(futures)
    for key, val in long_dict_:
      quan_dict.setdefault(key, []).append(val)

    #output sys dataframe
    quan_df = pd.DataFrame(quan_dict)
    quan_df.sort_values(by='quantiles', inplace=True)
    quan_df.set_index('quantiles').to_csv(f'results/{whichcat_deep}_{whichcat}.csv')    
    #quan_df.to_csv(f'results/{whichcat_deep}_{whichcat}.csv')    
