from coffea.util import load
from statsmodels.stats.weightstats import DescrStatsW
from collections import defaultdict
import ROOT as r 
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

years = ['2016preVFP']


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
theory_total_weight["weight"] = df["weight"].sum()
theory_total_weight["weight_scalep5p5"] = df["weight_scalep5p5"].sum()
theory_total_weight["weight_scale22"] = df["weight_scale22"].sum()
for i in range(103):
  theory_total_weight[f"weight_lhe{i}"] = df[f"weight_lhe{i}"].sum()

#Separate into two dataframe: one for VBF cat and GG cat
df_ggcat, df_vbfcat = df[(df['isVBFcat']==0)], df[(df['isVBFcat']==1)]

#Repeat for data
var_dict_data = {}
for year in years:
  print(f'Processing {year}')
  result = load(f"results/{year}/makeData/output_test.coffea")
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

mass = r.RooRealVar("CMS_emu_Mass", "m_{e#mu}", 110, 160, "GeV")
mva_dummy = r.RooRealVar("mva", "mva", 0, 1)
weight_dummy = r.RooRealVar("weight", "weight", -20, 20)
mva_sys_dummy = {}
for sys in leptonUnc:
  for UpDown in ['Up', 'Down']:
    mva_sys_dummy[f'mva_{sys}_{UpDown}'] = r.RooRealVar(f'mva_{sys}_{UpDown}', "mva", 0, 1)
e_m_Mass, weight, e_m_Mass_sys, weight_sys, mva, mva_sys = [], [], [], [], [], []

#Create dataframe for fast calulcations of systematics
for df_gg_vbf, df_gg_vbf_data, whichcat in zip([df_ggcat, df_vbfcat], [df_ggcat_data, df_vbfcat_data], ['GGcat', 'VBFcat']):
  #Get mva values corresponding to 100 quantiles 
  wq = DescrStatsW(data=df_gg_vbf['mva'], weights=df_gg_vbf['weight'])
  quantiles = wq.quantile(probs=np.linspace(0,1,101), return_pandas=False)

  #Create tree for mva/weight/e_m_Mass
  datasets = []
  tree_data = r.TTree("tree", "tree")
  #data
  e_m_Mass.append(df_gg_vbf_data[f'e_m_Mass'].to_numpy())
  mva.append(df_gg_vbf['mva'].to_numpy())
  tree_data.Branch("CMS_emu_Mass", e_m_Mass[-1], "CMS_emu_Mass/F")
  tree_data.Branch("mva", mva[-1], "mva/F")

  #Loop through quantiles and turn tree into RooDataSet
  for i in range(len(quantiles)-1):
      datasets.append(r.RooDataSet(f"data_norm_range{i+1}", f"data_norm_range{i+1}", tree_data, r.RooArgSet(mass,mva_dummy), f"(mva<{quantiles[i+1]}) && (mva>={quantiles[i]})"))

  #output all RooDataSet into workspace
  w = r.RooWorkspace("CMS_emu_workspace", "CMS_emu_workspace")
  for dataset in datasets:
    getattr(w, 'import')(dataset)
  w.Print()
  #fout = r.TFile(f"results/{whichcat}.root", "recreate")
  #w.writeToFile(f"results/{whichcat}.root")
  r.gDirectory.Add(w)

  quan_dict_gg, quan_dict_vbf = defaultdict(list), defaultdict(list)

  #Divide into GG/VBF
  for df_gg_vbf_deep, quan_dict, whichcat_deep in zip([df_gg_vbf[df_gg_vbf['isVBF']==0], df_gg_vbf[df_gg_vbf['isVBF']==1]], [quan_dict_gg, quan_dict_vbf], ['GG', 'VBF']):

    #Create tree for mva/weight/e_m_Mass/lep scale/smearing
    datasets = []
    tree = r.TTree("tree", "tree")

    #mc
    e_m_Mass.append(df_gg_vbf_deep[f'e_m_Mass'].to_numpy())
    weight.append(df_gg_vbf_deep['weight'].to_numpy())
    mva.append(df_gg_vbf_deep['mva'].to_numpy())
    tree.Branch("CMS_emu_Mass", e_m_Mass[-1], "CMS_emu_Mass/F")
    tree.Branch("weight", weight[-1], "weight/F")
    tree.Branch("mva", mva[-1], "mva/F")
    for sys in leptonUnc:
      for UpDown in ['Up', 'Down']:
        e_m_Mass_sys.append(df_gg_vbf_deep[f'e_m_Mass_{sys}_{UpDown}'].to_numpy())
        mva_sys.append(df_gg_vbf_deep[f'mva_{sys}_{UpDown}'].to_numpy())
        tree.Branch(f'CMS_emu_Mass_{sys}_{UpDown}', e_m_Mass_sys[-1], f"CMS_emu_Mass_{sys}_{UpDown}/F")
        tree.Branch(f"mva_{sys}_{UpDown}", mva_sys[-1], f"mva_{sys}_{UpDown}/F")

    #Loop through quantiles and turn tree into RooDataSet
    def FillSys(i):
      print(f"Doing quantile {i+1} for Siganl")
      datasets.append(r.RooDataSet(f"norm_range{i+1}", f"norm_range{i+1}", tree, r.RooArgSet(mass,mva_dummy,weight_dummy), f"(mva<{quantiles[i+1]}) && (mva>={quantiles[i]})", wgtVarName="weight"))
      for sys in leptonUnc:
        for UpDown in ['Up', 'Down']:
          datasets.append(r.RooDataSet(f"{sys}_{UpDown}_range{i+1}", f"{sys}_{UpDown}_range{i+1}", tree, r.RooArgSet(mass,mva_sys_dummy[f'mva_{sys}_{UpDown}'],weight_dummy), f"(mva_{sys}_{UpDown}<{quantiles[i+1]}) && (mva_{sys}_{UpDown}>={quantiles[i]})", wgtVarName="weight"))
      #Get the quantile dataframe
      subdf = df_gg_vbf_deep[(df_gg_vbf_deep['mva']<quantiles[i+1])&(df_gg_vbf_deep['mva']>=quantiles[i])]
    
      #Store all the sys weight/acceptance
      quan_dict['quantiles'].append(i+1)
      quan_dict['lowerMVA'].append(quantiles[i])
      quan_dict['weight'].append(subdf['weight'].sum())
      quan_dict["weight2016"].append(subdf[subdf["is2016"]==1]["weight"].sum())
      quan_dict["weight2017"].append(subdf[subdf["is2017"]==1]["weight"].sum())
      quan_dict["weight2018"].append(subdf[subdf["is2018"]==1]["weight"].sum())
      quan_dict['acc'].append(quan_dict['weight'][-1]/theory_total_weight['weight'])
    
      for sys in theoUnc:
        quan_dict[f'weight_{sys}'].append(subdf[f'weight_{sys}'].sum()/theory_total_weight[f'weight_{sys}'])
      for UpDown in ['Up', 'Down']:
        for sys in sfUnc:
          quan_dict[f'weight_{sys}_{UpDown}'].append(subdf[f'weight_{sys}_{UpDown}'].sum())
        for sys in leptonUnc+metUnc:
          subdf = df_gg_vbf_deep[(df_gg_vbf_deep[f'mva_{sys}_{UpDown}']<quantiles[i+1])&(df_gg_vbf_deep[f'mva_{sys}_{UpDown}']>=quantiles[i])]
          quan_dict[f'weight_{sys}_{UpDown}'].append(subdf[f'weight'].sum())
    
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
          quan_dict[f'weight_{sys}_{UpDown}'].append(subdf[f'weight'].sum())

    executor = concurrent.futures.ProcessPoolExecutor(32)
    futures = [executor.submit(FillSys, i) 
           for i in range(len(quantiles)-1)]
    concurrent.futures.wait(futures)
#    for i in range(len(quantiles)-1):
#      FillSys(i)

    #output all RooDataSet into workspace
    w = r.RooWorkspace("CMS_emu_workspace", "CMS_emu_workspace")
    for dataset in datasets:
      getattr(w, 'import')(dataset)
    w.Print()
    #fout = r.TFile(f"result/{whichcat_deep}_{whichcat}.root", "recreate")
    #w.writeToFile(f"results/{whichcat_deep}_{whichcat}.root")
    r.gDirectory.Add(w)

    #output sys dataframe
    quan_dict = pd.DataFrame(quan_dict)
    #quan_dict.set_index('quantiles').to_csv(f'results/{whichcat_deep}_{whichcat}.csv')    
