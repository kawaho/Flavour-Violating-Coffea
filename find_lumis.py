import os, glob, json, uproot

datadefs = {}

datadefs['GluGlu_LFV_HToEMu'] = 48.58*0.01

datadefs['VBF_LFV_HToEMu'] = 3.782*0.01

datadefs['DYJetsToLL_M-50'] = 5398.0 #5343.0,

datadefs['DYJetsToLL_M-10to50'] = 15890.0 #18610.0,

datadefs['DYJetsToLL_0J'] = 928.3 #877.8,

datadefs['DYJetsToLL_1J'] = 292.4 #325.7, #Not on XSDB304.4,

datadefs['DYJetsToLL_2J'] = 86.53 #111.5,

datadefs['WJetsToLNu_TuneCP5'] = 53870.0 #52940.0,

datadefs['"WJetsToLNu_0J'] = 8927.0 #8104.0,

datadefs['WJetsToLNu_1J'] = 2809.0 #2793.0,

datadefs['WJetsToLNu_2J'] = 826.3 #992.5,

datadefs['WGToLNuG_TuneCP5'] = 464.4

datadefs['ZZ_TuneCP5'] = 12.17 #12.14,

datadefs['WZ_TuneCP5'] = 27.59 #27.57,

datadefs['WW_TuneCP5'] = 75.95 #75.88,

datadefs['TTTo2L2Nu'] = 88.29

datadefs['TTToSemiLeptonic'] = 365.34

datadefs['TTToHadronic'] = 377.96

datadefs['ST_tW_antitop_5f_inclusiveDecays'] = 35.85

datadefs['ST_tW_top_5f_inclusiveDecays'] = 35.85

datadefs['ST_t-channel_antitop_4f_inclusiveDecays'] = 80.95

datadefs['ST_t-channel_top_4f_inclusiveDecays'] = 136.02

datadefs['EWKZ2Jets_ZToLL'] = 3.987

datadefs['EWKZ2Jets_ZToNuNu'] = 10.01

def mclumi(sample):
  for i in datadefs:
    if i in sample:
      return i, datadefs[i]
  return -1, -1

if __name__ == '__main__':
  datalumis = {'2016preVFP': 36330, '2016postVFP': 36330, '2017': 41476.1, '2018': 59830}
  year = '2017'
  if True: #for year in datalumis:
    samples_names = glob.glob('/hdfs/store/user/kaho/NanoPost_'+year+'/*')
    sample_paths = {}
    lumiWeight = {}
    for name in samples_names:
       if 'SingleMuon' in name: continue
       sample_basename = os.path.basename(name)
       sample_paths[sample_basename] = glob.glob(name+'/*/*/*/*root')
       lumiWeight[mclumi(sample_basename)[0]] = 0
       runTrees = [i+':Runs' for i in sample_paths[sample_basename]]
       for runTree in uproot.iterate(runTrees, ['genEventSumw'], num_workers=10):
         lumiWeight[mclumi(sample_basename)[0]]+=sum(runTree['genEventSumw']) 
   
    for sample_basename in lumiWeight:
      lumiWeight[mclumi(sample_basename)[0]] = mclumi(sample_basename)[1]*datalumis[year]/lumiWeight[mclumi(sample_basename)[0]]  

    with open('lumi_'+year+'.json', 'w') as f: 
      json.dump(lumiWeight, f, indent=4)
      f.close()
