from coffea.util import load
import coffea.hist
from statsmodels.stats.weightstats import DescrStatsW
from collections import defaultdict
import uproot
import numpy as np
import pandas as pd
import glob, os, json, argparse
import concurrent.futures

#Listing all systematics
metUnc = ['UnclusteredEn']
jetUnc = ['jesAbsolute', 'jesBBEC1', 'jesFlavorQCD', 'jesEC2', 'jesHF', 'jesRelativeBal']
jetyearUnc = sum([[f'jer_{year}', f'jesAbsolute_{year}', f'jesBBEC1_{year}', f'jesEC2_{year}', f'jesHF_{year}', f'jesRelativeSample_{year}'] for year in ['2017', '2018', '2016']], [])
sfUnc = sum([[f'pu_{year}', f'bTag_{year}'] for year in ['2017', '2018', '2016']], [])
sfUnc += ['pf_2016', 'pf_2017', 'mID', 'mIso', 'mTrg', 'eTrig', 'eIso', 'eReco', 'eID']
theoUnc = [f'lhe{i}' for i in range(103)] + ['scalep5p5', 'scale22']
leptonUnc = ['ess', 'me']#'ees', 'eer'

years = ['2016preVFP', '2016postVFP', '2017', '2018']
BRcorr = {120:{'GG':45.14/52.22, 'VBF':4.086/3.935}, 125:{'GG':41.98/48.61, 'VBF':3.925/3.766}, 130:{'GG':39.14/45.31, 'VBF':3.773/3.637}}

for masspt in [125]:#, 110, 120, 130, 140, 150, 160]:  
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
    for cat in cats:
      weight_year[cat].append(result[f'mva_hist-{cat}'].values()[()])
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
  
  #Get total weight of all years for acceptance calculations
  theory_total_weight = {}
  df = pd.DataFrame(var_dict)
  for c in ['GG', 'VBF']:
    theory_total_weight[f"weight_{c}"] = sum(hist_dict[f'mva_hist-{c}_VBFcat'])+sum(hist_dict[f'mva_hist-{c}_GGcat'])
    theory_total_weight[f"weight_scalep5p5_{c}"] = sum(hist_dict[f'mva_hist_scalep5p5-{c}_VBFcat'])+sum(hist_dict[f'mva_hist_scalep5p5-{c}_GGcat'])
    theory_total_weight[f"weight_scale22_{c}"] = sum(hist_dict[f'mva_hist_scale22-{c}_VBFcat'])+sum(hist_dict[f'mva_hist_scale22-{c}_GGcat'])
    for i in range(103):
      theory_total_weight[f"weight_lhe{i}_{c}"] = sum(hist_dict[f'mva_hist_lhe{i}-{c}_VBFcat'])+sum(hist_dict[f'mva_hist_lhe{i}-{c}_GGcat'])
  
  
  #Separate into two dataframe: one for VBF cat and GG cat
  df_ggcat, df_vbfcat = df[(df['isVBFcat']==0)&(df['isHerwig']==0)], df[(df['isVBFcat']==1)&(df['isHerwig']==0)]
  
  #Create dataframe for fast calulcations of systematics
  for df_gg_vbf, whichcat in zip([df_ggcat, df_vbfcat], ['GGcat', 'VBFcat']):
    print(f'Running {whichcat}')
    #Get mva values corresponding to 100 quantiles 
    quantiles = np.arange(0,1.01,0.01) #np.load(f"results/SenScan_v9/{whichcat}_quantiles",allow_pickle=True)

    #Divide into GG/VBF
    for df_gg_vbf_deep, whichcat_deep in zip([df_gg_vbf[df_gg_vbf['isVBF']==0], df_gg_vbf[df_gg_vbf['isVBF']==1]], ['GG', 'VBF']):
      print(f'Running {whichcat_deep} {masspt}')
      quan_dict = defaultdict(list)
  
      #Create tree for mva/weight/e_m_Mass/lep scale/smearing
      fsig = uproot.recreate(f"results/SenScan_v9/{whichcat_deep}_{whichcat}_{masspt}_tree.root")
      #mc
#histogram
#      h = [] 
#      e_m_Mass = df_gg_vbf_deep[f'e_m_Mass'].to_numpy()
#      weight = df_gg_vbf_deep['weight'].to_numpy()
#      mva = df_gg_vbf_deep['mva'].to_numpy()
#
#      h.append(coffea.hist.Hist("nominal",
#                     coffea.hist.Bin("e_m_Mass", "e_m_Mass", 70000, 100, 170),
#                     coffea.hist.Bin("mva", "mva", 100, 0, 1),
#                     ))
#      fillContent = {'CMS_emu_Mass': e_m_Mass, 'mva': mva, 'weight': weight}
#      h[-1].fill(**fillContent)
#      fsig['nominal'] = h[-1].to_hist()

      groups = df_gg_vbf_deep.groupby(pd.cut(df_gg_vbf_deep.mva, quantiles))

      for i, group in enumerate(groups):
        if masspt in [120,125,130]:
          fsig[f'tree_{i+1}'] = {'CMS_emu_Mass': group[1].e_m_Mass.to_numpy(), 'weight': group[1]['weight'].to_numpy()*BRcorr[masspt][whichcat_deep]}
        else:
          fsig[f'tree_{i+1}'] = {'CMS_emu_Mass': group[1].e_m_Mass.to_numpy(), 'weight': group[1]['weight'].to_numpy()}
#      e_m_Mass_sys, mva_sys = [], []
      for sys in leptonUnc:
        for UpDown in ['Up', 'Down']:
#          e_m_Mass_sys.append(df_gg_vbf_deep[f'e_m_Mass_{sys}_{UpDown}'].to_numpy())
#          mva_sys.append(df_gg_vbf_deep[f'mva_{sys}_{UpDown}'].to_numpy())
#          h.append(coffea.hist.Hist(f"{sys}_{UpDown}",
#                         coffea.hist.Bin("e_m_Mass", "e_m_Mass", 70000, 100, 170),
#                         coffea.hist.Bin("mva", "mva", 100, 0, 1),
#                         ))
#          fillContent = {'CMS_emu_Mass': e_m_Mass_sys[-1], 'mva': mva_sys[-1], 'weight': weight}
#          h[-1].fill(**fillContent)
#          fsig[f'{sys}_{UpDown}'] = h[-1].to_hist()

          groups = df_gg_vbf_deep.groupby(pd.cut(df_gg_vbf_deep[f'mva_{sys}_{UpDown}'], quantiles))

          for i, group in enumerate(groups):
            fsig[f'tree_{sys}_{UpDown}_{i+1}'] = {'CMS_emu_Mass': group[1][f'e_m_Mass_{sys}_{UpDown}'].to_numpy(), 'weight': group[1]['weight'].to_numpy()}


      #Loop through quantiles and turn tree into RooDataSet
  
      dict_ = {}
      dict_['quantiles'] = np.arange(1,101) 
      dict_['lowerMVA'] = quantiles[:-1]
      dict_['weight'] = hist_dict[f'mva_hist-{whichcat_deep}_{whichcat}']
      dict_['weight_herwig'] = hist_dict[f'mva_hist-herwig_{whichcat}']
      dict_["weight2016"] = weight_year[f'{whichcat_deep}_{whichcat}'][0]+weight_year[f'{whichcat_deep}_{whichcat}'][1]
      dict_["weight2017"] = weight_year[f'{whichcat_deep}_{whichcat}'][2]
      dict_["weight2018"] = weight_year[f'{whichcat_deep}_{whichcat}'][3]
      dict_['acc'] = dict_['weight']/theory_total_weight[f'weight_{whichcat_deep}']
      for sys in theoUnc:
        dict_[f'weight_{sys}'] = hist_dict[f'mva_hist_{sys}-{whichcat_deep}_{whichcat}']/theory_total_weight[f'weight_{sys}_{whichcat_deep}']
      for UpDown in ['Up', 'Down']:
        for sys in jetUnc+jetyearUnc+sfUnc+leptonUnc+metUnc:
          dict_[f'weight_{sys}_{UpDown}'] = hist_dict[f'mva_hist_{sys}_{UpDown}-{whichcat_deep}_{whichcat}']
      
      #output sys dataframe
      quan_df = pd.DataFrame.from_dict(dict_)
      quan_df.sort_values(by='quantiles', inplace=True)
      quan_df.set_index('quantiles').to_csv(f'results/SenScan_v9/{whichcat_deep}_{whichcat}_{masspt}.csv')    
