import json, glob
samples = {}
years = ['2016preVFP', '2016postVFP', '2017', '2018']
samples_to_find = ['data', 'GluGlu_LFV_HToEMu_M125', 'VBF_LFV_HToEMu_M125', 'TTTo2L2Nu', 'GluGlu_LFV_HToEMu_M120', 'VBF_LFV_HToEMu_M120', 'GluGlu_LFV_HToEMu_M130', 'VBF_LFV_HToEMu_M130', 'ZZ_TuneCP5', 'WZ_TuneCP5', 'WW_TuneCP5', 'DYJetsToLL_M-50', 'DYJetsToLL_M-10to50', 'DYJetsToLL_0J', 'DYJetsToLL_1J', 'DYJetsToLL_2J', 'WGToLNuG_TuneCP5', 'TTToSemiLeptonic', 'TTToHadronic', 'ST_tW_antitop_5f_inclusiveDecays', 'ST_tW_top_5f_inclusiveDecays', 'ST_t-channel_antitop_4f_InclusiveDecays', 'ST_t-channel_top_4f_InclusiveDecays', 'EWKZ2Jets_ZToLL', 'EWKZ2Jets_ZToNuNu', 'EWKWMinus2Jets', 'EWKWPlus2Jets', 'GluGluHToTauTau', 'VBFHToTauTau', 'WminusHToTauTau', 'WplusHToTauTau', 'GluGluHToWWTo2L2Nu', 'VBFHToWWTo2L2Nu', 'ZHToTauTau', 'W1JetsToLNu', 'W2JetsToLNu', 'W3JetsToLNu', 'W4JetsToLNu', 'WJetsToLNu_TuneCP5', '"WJetsToLNu_0J', 'WJetsToLNu_1J', 'WJetsToLNu_2J']
for year in years:
  for samples_shorthand in samples_to_find:
    #samples[samples_shorthand] = [i.replace('/hdfs', 'root://ndcms.crc.nd.edu/') for i in glob.glob(f'/hdfs/store/user/kaho/NanoPost_{year}_v2/{samples_shorthand}*/*/*/*/*root')]
  #samples['data'] = [i.replace('/hdfs', 'root://ndcms.crc.nd.edu/') for i in glob.glob(f'/hdfs/store/user/kaho/NanoPost_{year}_v2/SingleMuon/*/*/*/*root')]
    samples[samples_shorthand] = [i.replace('/hdfs', 'root://cmsxrootd.hep.wisc.edu/') for i in glob.glob(f'/hdfs/store/user/kaho/NanoPost_{year}_v2/{samples_shorthand}*/*/*/*/*root')]
  samples['data'] = [i.replace('/hdfs', 'root://cmsxrootd.hep.wisc.edu/') for i in glob.glob(f'/hdfs/store/user/kaho/NanoPost_{year}_v2/SingleMuon/*/*/*/*root')]
  with open('samples_'+year+'.json', 'w') as f: 
    json.dump(samples, f, indent=4)
    f.close()
