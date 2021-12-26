import os, glob, json, re, uproot

datadefs = {}

#https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV
#https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNHLHE2019
#m_{H} = 125.09GeV
datadefs['GluGlu_LFV_HToEMu_M125'] = 48.61*0.01
datadefs['VBF_LFV_HToEMu_M125'] = 3.766*0.01
datadefs['GluGlu_LFV_HToEMu_M120'] = datadefs['GluGlu_LFV_HToEMu_M125']
datadefs['GluGlu_LFV_HToEMu_M130'] = datadefs['GluGlu_LFV_HToEMu_M125']
datadefs['VBF_LFV_HToEMu_M120'] = datadefs['VBF_LFV_HToEMu_M125']
datadefs['VBF_LFV_HToEMu_M130'] = datadefs['VBF_LFV_HToEMu_M125']
datadefs['herwig7'] = datadefs['VBF_LFV_HToEMu_M125']

datadefs['GluGluHToTauTau'] = 48.61*6.256e-02
datadefs['VBFHToTauTau'] = 3.766*6.256e-02

datadefs['WminusHToTauTau'] = 5.27e-01*6.256e-02
datadefs['WplusHToTauTau'] = 8.31e-01*6.256e-02

datadefs['GluGluHToWWTo2L2Nu'] = 48.61*2.152e-01*(0.1086*3)**2
datadefs['VBFHToWWTo2L2Nu'] = 3.766*2.152e-01*(0.1086*3)**2

datadefs['ZHToTauTau'] = 8.80e-01*6.256e-02

#https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
#>50 6077.22 +-1.49 (integration) +- 14.78 (pdf) +- 2% (scale) 
datadefs['DYJetsToLL_M-50'] = 6077.22 #6404.0 #+-27.69 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2
datadefs['DYJetsToLL_0J'] = 5129.0 #+-8.715 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 
datadefs['DYJetsToLL_1J'] = 951.5 #+-6.067 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1
datadefs['DYJetsToLL_2J'] = 361.4 #+-3.704 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1 

#lfv NNLO
datadefs['DYJetsToLL_M-10to50'] = 18610.0 #15890.0+-24.94 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO

#NLO 
#61526.7 NNLO Htt
#https://twiki.cern.ch/twiki/bin/viewauth/CMS/HowToGenXSecAnalyzer#Running_the_GenXSecAnalyzer_on_a
#datadefs['WJetsToLNu_TuneCP5'] = 53870.0 #TODO GenXSecAnalyzer
datadefs['WJetsToLNu_0J'] = 53330.0 #+-90.89 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 NLO
datadefs['WJetsToLNu_1J'] = 8875.0 #+-55.31 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1 NLO
datadefs['WJetsToLNu_2J'] = 3338.0 #+-34.64 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1 NLO

#LO
datadefs['WJetsToLNu_TuneCP5'] = 53870.0 #+-129.7 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2
datadefs['W1JetsToLNu'] = 8927.0 #+-24.09 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1
datadefs['W2JetsToLNu'] = 2809.0 #+-8.201 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1
datadefs['W3JetsToLNu'] = 826.3 #+-2.511 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v1 
datadefs['W4JetsToLNu'] = 388.3 #+-0.3804 GenXSecAnalyzer 

datadefs['WGToLNuG_TuneCP5'] = 412.7 #+-1.027 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO

#https://twiki.cern.ch/twiki/bin/viewauth/CMS/StandardModelCrossSectionsat13TeV
datadefs['WW_TuneCP5'] = 118.7 #+2.5%-2.2% NNLO

#lfv NNLO
#https://indico.cern.ch/event/783842/contributions/3376363/attachments/1897033/3130087/DiBosons-Calderon_v2.pdf
datadefs['ZZ_TuneCP5'] = 16.91 #12.17+-0.01966 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2
datadefs['WZ_TuneCP5'] = 49.98 #27.59+-0.03993 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2

#XSDB
#TTbar inclusive: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
#W BR: https://pdg.lbl.gov/2021/tables/contents_tables.html
#https://wiki.physik.uzh.ch/cms/physics:crosssections
datadefs['TTTo2L2Nu'] = 88.29 #+4.8%-6.1% NNLO
datadefs['TTToSemiLeptonic'] = 365.34 #+4.8%-6.1% NNLO
datadefs['TTToHadronic'] = 377.96 #+4.8%-6.1% NNLO

#https://twiki.cern.ch/twiki/bin/view/LHCPhysics/SingleTopRefXsec
datadefs['ST_tW_antitop_5f_inclusiveDecays'] = 35.85 #71.7/2 Scale: +1.80-1.80 PDF: +3.40-3.40 NLO
datadefs['ST_tW_top_5f_inclusiveDecays'] = 35.85 #see above 
datadefs['ST_t-channel_antitop_4f_InclusiveDecays'] = 80.95 #+4.06-3.61 NLO
datadefs['ST_t-channel_top_4f_InclusiveDecays'] = 136.02 #+5.40-4.57 NLO

datadefs['EWKZ2Jets_ZToLL'] = 6.215 #+-0.004456 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO

datadefs['EWKZ2Jets_ZToNuNu'] = 10.66 #+-2.969e-03 GenXSecAnalyzer

datadefs['EWKWMinus2Jets'] = 32.05 #+-0.02492 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO
datadefs['EWKWPlus2Jets'] = 39.05 #+-0.0291 RunIISummer20UL16MiniAOD-106X_mcRun2_asymptotic_v13-v2 LO

DYNNLO = 6077.22/datadefs['DYJetsToLL_M-50']
WNNLO = 61526.7/datadefs['WJetsToLNu_TuneCP5']
dyLumi, wLumi = 1, 1


def mclumi(sample):
  for i in datadefs:
    if i in sample:
      return i, datadefs[i]
  print("wrong", sample)
  return -1, -1

if __name__ == '__main__':
  datalumis = {'2016preVFP': 19514.7, '2016postVFP': 16810.8, '2017': 41476.1, '2018': 59817.3}
  for year in ['2016postVFP', '2017', '2018']: #['2016preVFP']:#datalumis:
    samples_names = glob.glob('/hdfs/store/user/kaho/NanoPost_'+year+'_v2/*')
    sample_paths = {}
    lumiWeight = {}
    for name in samples_names:
       print(f"Running sample {name}")
       if 'SingleMuon' in name or 'tmp' in name or 'WJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-pythia8' in name: continue
       sample_basename = os.path.basename(name)
       sample_paths[sample_basename] = glob.glob(name+'/*/*/*/*root')
       lumiWeight[mclumi(sample_basename)[0]] = 0
       runTrees = [i+':Runs' for i in sample_paths[sample_basename]]
       for runTree in uproot.iterate(runTrees, ['genEventSumw'], num_workers=12):
         lumiWeight[mclumi(sample_basename)[0]]+=sum(runTree['genEventSumw']) 
  
    for sample_basename in lumiWeight:
      if (re.match(re.compile('W.Jet'), sample_basename)) or "WJetsToLNu_TuneCP5" in sample_basename: 
      #if ('DY' in sample_basename and not '10to50' in sample_basename) or (re.match(re.compile('W.Jet'), sample_basename)) or "WJetsToLNu_TuneCP5" in sample_basename: 
        print(f"{sample_basename} is DYMll50 or W+Jets")
        lumiWeight[sample_basename] = lumiWeight[sample_basename]/(mclumi(sample_basename)[1])
        if sample_basename=="DYJetsToLL_M-50":
          dyLumi = lumiWeight[sample_basename]
        if sample_basename=="WJetsToLNu_TuneCP5":
          wLumi = lumiWeight[sample_basename]

      else:
        lumiWeight[sample_basename] = mclumi(sample_basename)[1]*datalumis[year]/lumiWeight[sample_basename]  

    for sample_basename in lumiWeight:
      if "WJetsToLNu_TuneCP5" in sample_basename: 
        lumiWeight[sample_basename] = WNNLO*datalumis[year]/wLumi
      elif re.match(re.compile('W.Jet'), sample_basename): 
        lumiWeight[sample_basename] = WNNLO*datalumis[year]/(lumiWeight[sample_basename]+wLumi) 
      #elif "DYJetsToLL_M-50" in sample_basename:
      #  lumiWeight[sample_basename] = DYNNLO*datalumis[year]/dyLumi
      #elif "DY" in sample_basename and not "M-10to50" in sample_basename:
      #  lumiWeight[sample_basename] = DYNNLO*datalumis[year]/(lumiWeight[sample_basename]+dyLumi)

    with open('lumi_'+year+'.json', 'w') as f: 
      json.dump(lumiWeight, f, indent=4)
      f.close()
