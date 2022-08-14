import numpy as np
import pandas as pd
import xgboost as xgb 
import uproot
from time import perf_counter

filters = [('label', '==', 3), ('ept', '>', 25), ('mpt', '>', 20)]

for whichcat in ['GGcat', 'VBFcat']:
 
  #Get model
  whichcat2 = whichcat.replace('cat','').lower()
  model = xgb.XGBClassifier()
  model.load_model(f'XGBoost-for-HtoEMu/results/model_{whichcat2}_v9.json')
  feature_names = model.get_booster().feature_names

  #tmp method to remove variables not pre-calculated
  features = list(feature_names)
  if 'empt_Per_e_m_Mass' in feature_names: features.remove('empt_Per_e_m_Mass')
    
  #Load df
  data_vbf = pd.read_parquet(f'results/csv4BDT/makeDF_v9_diboson_others_tt_signal_signalAlt_data_oc_{whichcat2}.parquet', engine='pyarrow', filters=filters, columns=features+['empt', 'e_m_Mass'])

  #Calculate extra variables
  data_vbf['empt_Per_e_m_Mass'] = data_vbf['empt']/data_vbf['e_m_Mass']

  #Get MVA scores
  data_vbf['mva'] = model.predict_proba(data_vbf[feature_names])[:,1]
  
  #Group data by quantiles
  quantiles = np.arange(0,1.01,0.01)
  #Create tree for mva/weight/e_m_Mass
  groups = data_vbf.groupby(pd.cut(data_vbf.mva, quantiles))
  
  with uproot.recreate(f"results/SenScan_v9/{whichcat}_tree_test.root") as f:
    for i, group in enumerate(groups):
      f[f'tree_{i+1}'] = {'CMS_emu_Mass': group[1].e_m_Mass.to_numpy()}
