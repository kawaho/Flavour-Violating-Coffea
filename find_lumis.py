import os, glob, json, uproot

DYNNLO = 6077.22/6404.0
WNNLO = 61526.7/53870.0
dyLumi, wLumi = 1, 1

datadefs = {}

#https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV
#m_{H} = 125.09GeV
datadefs['GluGlu_LFV_HToEMu_M125'] = 48.52*0.01
datadefs['VBF_LFV_HToEMu_M125'] = 3.779*0.01
datadefs['GluGlu_LFV_HToEMu_M120'] = datadefs['GluGlu_LFV_HToEMu_M125']
datadefs['GluGlu_LFV_HToEMu_M130'] = datadefs['GluGlu_LFV_HToEMu_M125']
datadefs['VBF_LFV_HToEMu_M120'] = datadefs['VBF_LFV_HToEMu_M125']
datadefs['VBF_LFV_HToEMu_M130'] = datadefs['VBF_LFV_HToEMu_M125']
datadefs['herwig7'] = datadefs['VBF_LFV_HToEMu_M125']

datadefs['GluGluHToTauTau'] = 48.52*6.256e-02
datadefs['VBFHToTauTau'] = 3.779*6.256e-02
datadefs['WminusHToTauTau'] = 5.313e-01*6.256e-02
datadefs['WplusHToTauTau'] = 8.380e-01*6.256e-02

datadefs['GluGluHToWWTo2L2Nu'] = 48.52*2.152e-01*0.1086**2
datadefs['VBFHToWWTo2L2Nu'] = 3.779*2.152e-01*0.1086**2

datadefs['ZHToTauTau'] = 8.824e-01*6.256e-02

#https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
#>50 6077.22 +-1.49 (integration) +- 14.78 (pdf) +- 2% (scale) 
datadefs['DYJetsToLL_M-50'] = 6404.0 #+-27.69 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2
datadefs['DYJetsToLL_0J'] = 5129.0 #+-8.715 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 
datadefs['DYJetsToLL_1J'] = 951.5 #+-6.067 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1
datadefs['DYJetsToLL_2J'] = 361.4 #+-3.704 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1 

#lfv NNLO
datadefs['DYJetsToLL_M-10to50'] = 18610.0 #15890.0+-24.94 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO

#61526.7 NNLO Htt
#https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToGenXSecAnalyzer#Running_the_GenXSecAnalyzer_on_a
datadefs['WJetsToLNu_TuneCP5'] = 53870.0 #TODO GenXSecAnalyzer
datadefs['WJetsToLNu_0J'] = 53330.0 #+-90.89 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 NLO
datadefs['WJetsToLNu_1J'] = 8875.0 #+-55.31 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1 NLO
datadefs['WJetsToLNu_2J'] = 3338.0 #+-34.64 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1 NLO

datadefs['WGToLNuG_TuneCP5'] = 412.7 #+-1.027 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO

#https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
datadefs['WW_TuneCP5'] = 118.7 #+2.5%-2.2% NNLO

#lfv NNLO
datadefs['ZZ_TuneCP5'] = 16.91 #12.17+-0.01966 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2
datadefs['WZ_TuneCP5'] = 51.11 #27.59+-0.03993 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2

#XSDB
datadefs['TTTo2L2Nu'] = 88.29 #+4.8%-6.1% NNLO
datadefs['TTToSemiLeptonic'] = 365.34 #+4.8%-6.1% NNLO
datadefs['TTToHadronic'] = 377.96 #+4.8%-6.1% NNLO

#https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopRefXsec
datadefs['ST_tW_antitop_5f_inclusiveDecays'] = 35.85 #71.7/2 Scale: +1.80-1.80 PDF: +3.40-3.40 NLO
datadefs['ST_tW_top_5f_inclusiveDecays'] = 35.85 #see above 
datadefs['ST_t-channel_antitop_5f_InclusiveDecays'] = 80.95 #+4.06-3.61 NLO
datadefs['ST_t-channel_top_5f_InclusiveDecays'] = 136.02 #+5.40-4.57 NLO

datadefs['EWKZ2Jets_ZToLL'] = 6.215 #+-0.004456 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO

datadefs['EWKZ2Jets_ZToNuNu'] = 10.66 #+-2.969e-03 GenXSecAnalyzer

datadefs['EWKWMinus2Jets'] = 32.05 #+-0.02492 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO
datadefs['EWKWPlus2Jets'] = 39.05 #+-0.0291 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO


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
      if ('DY' in sample_basename and not '10to50' in sample_basename) or ('WJ' in sample_basename): 
        lumiWeight[sample_basename] = lumiWeight[sample_basename]/(mclumi(sample_basename)[1])
        if sample_basename=="DYJetsToLL_M-50":
          dyLumi = lumiWeight[sample_basename]
        if sample_basename=="WJetsToLNu_TuneCP5":
          wLumi = lumiWeight[sample_basename]

      else:
        lumiWeight[sample_basename] = mclumi(sample_basename)[1]*datalumis[year]/lumiWeight[sample_basename]  

    for sample_basename in lumiWeight:
      if "DYJetsToLL_M-50" in sample_basename:
        lumiWeight[sample_basename] = DYNNLO*datalumis[year]/dyLumi
      elif "DY" in sample_basename and not "M-10to50" in sample_basename:
        lumiWeight[sample_basename] = DYNNLO*datalumis[year]/(lumiWeight[sample_basename]+dyLumi)
      elif "WJetsToLNu_TuneCP5" in sample_basename: 
        lumiWeight[sample_basename] = WNNLO*datalumis[year]/wLumi
      elif "WJ" in sample_basename:
        lumiWeight[sample_basename] = WNNLO*datalumis[year]/(lumiWeight[sample_basename]+wLumi) 

    with open('lumi_'+year+'.json', 'w') as f: 
      json.dump(lumiWeight, f, indent=4)
      f.close()
