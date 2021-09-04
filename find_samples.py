samples_to_run = {
'signal': {'GluGlu_LFV_HToEMu_M125', 'VBF_LFV_HToEMu_M125'},
'data': {'SingleMuon'},
'diboson': {'ZZ_TuneCP5', 'WZ_TuneCP5', 'WW_TuneCP5'},
'top': {'TTTo2L2Nu', 'TTToSemiLeptonic', 'TTToHadronic', 'ST_tW_antitop_5f_inclusiveDecays', 'ST_tW_top_5f_inclusiveDecays', 'ST_t-channel_antitop_5f_inclusiveDecays', 'ST_t-channel_top_5f_InclusiveDecays'},
'dy': {'DYJetsToLL_M-50', 'DYJetsToLL_M-10to50', 'DYJetsToLL_0J', 'DYJetsToLL_1J', 'DYJetsToLL_2J'},
'wjets': {'WJetsToLNu_TuneCP5', '"WJetsToLNu_0J', 'WJetsToLNu_1J', 'WJetsToLNu_2J'},'higgs': {'GluGluHToTauTau', 'VBFHToTauTau', 'WminusHToTauTau', 'WplusHToTauTau', 'GluGluHToWWTo2L2Nu', 'VBFHToWWTo2L2Nu', 'ZHToTauTau'},
'ewk': {'EWKWMinus2Jets', 'EWKWPlus2Jets', 'EWKZ2Jets_ZToLL', 'EWKZ2Jets_ZToNuNu', 'WGToLNuG_TuneCP5'},

'makeHist': ['GluGlu_LFV_HToEMu_M125', 'VBF_LFV_HToEMu_M125', 'data', 'ZZ_TuneCP5', 'WZ_TuneCP5', 'WW_TuneCP5', 'TTTo2L2Nu', 'DYJetsToLL_M-50', 'DYJetsToLL_M-10to50', 'DYJetsToLL_0J', 'DYJetsToLL_1J', 'DYJetsToLL_2J', 'WGToLNuG_TuneCP5', 'TTToSemiLeptonic', 'TTToHadronic', 'ST_tW_antitop_5f_inclusiveDecays', 'ST_tW_top_5f_inclusiveDecays', 'ST_t-channel_antitop_5f_inclusiveDecays', 'ST_t-channel_top_5f_InclusiveDecays', 'EWKZ2Jets_ZToLL', 'EWKZ2Jets_ZToNuNu', 'EWKWMinus2Jets', 'EWKWPlus2Jets', 'GluGluHToTauTau', 'VBFHToTauTau', 'WminusHToTauTau', 'WplusHToTauTau', 'GluGluHToWWTo2L2Nu', 'VBFHToWWTo2L2Nu', 'ZHToTauTau', 'WJetsToLNu_TuneCP5', 'W1JetsToLNu', 'W2JetsToLNu', 'W3JetsToLNu', 'W4JetsToLNu'], #, 'WJetsToLNu_TuneCP5', '"WJetsToLNu_0J', 'WJetsToLNu_1J', 'WJetsToLNu_2J'

'makeDF': ['GluGlu_LFV_HToEMu_M125'], #'VBF_LFV_HToEMu_M125', 'ZZ_TuneCP5', 'WZ_TuneCP5', 'WW_TuneCP5', 'TTTo2L2Nu', 'DYJetsToLL_M-50', 'DYJetsToLL_M-10to50', 'DYJetsToLL_0J', 'DYJetsToLL_1J', 'DYJetsToLL_2J', 'WGToLNuG_TuneCP5', 'TTToSemiLeptonic', 'TTToHadronic', 'ST_tW_antitop_5f_inclusiveDecays', 'ST_tW_top_5f_inclusiveDecays', 'ST_t-channel_antitop_5f_inclusiveDecays', 'ST_t-channel_top_5f_InclusiveDecays', 'EWKZ2Jets_ZToLL', 'EWKZ2Jets_ZToNuNu', 'EWKWMinus2Jets', 'EWKWPlus2Jets', 'GluGluHToTauTau', 'VBFHToTauTau', 'WminusHToTauTau', 'WplusHToTauTau', 'GluGluHToWWTo2L2Nu', 'VBFHToWWTo2L2Nu', 'ZHToTauTau', 'WJetsToLNu_TuneCP5', 'W1JetsToLNu', 'W2JetsToLNu', 'W3JetsToLNu', 'W4JetsToLNu'], #, 'WJetsToLNu_TuneCP5', '"WJetsToLNu_0J', 'WJetsToLNu_1J', 'WJetsToLNu_2J'

'bTagWP': ['GluGlu_LFV_HToEMu_M125', 'VBF_LFV_HToEMu_M125', 'data', 'ZZ_TuneCP5', 'WZ_TuneCP5', 'WW_TuneCP5', 'TTTo2L2Nu', 'DYJetsToLL_M-50', 'DYJetsToLL_M-10to50', 'DYJetsToLL_0J', 'DYJetsToLL_1J', 'DYJetsToLL_2J', 'WGToLNuG_TuneCP5', 'TTToSemiLeptonic', 'TTToHadronic', 'ST_tW_antitop_5f_inclusiveDecays', 'ST_tW_top_5f_inclusiveDecays', 'ST_t-channel_antitop_5f_inclusiveDecays', 'ST_t-channel_top_5f_InclusiveDecays', 'EWKZ2Jets_ZToLL', 'EWKZ2Jets_ZToNuNu', 'EWKWMinus2Jets', 'EWKWPlus2Jets', 'GluGluHToTauTau', 'VBFHToTauTau', 'WminusHToTauTau', 'WplusHToTauTau', 'GluGluHToWWTo2L2Nu', 'VBFHToWWTo2L2Nu', 'ZHToTauTau', 'WJetsToLNu_TuneCP5', 'W1JetsToLNu', 'W2JetsToLNu', 'W3JetsToLNu', 'W4JetsToLNu'], #, 'WJetsToLNu_TuneCP5', '"WJetsToLNu_0J', 'WJetsToLNu_1J', 'WJetsToLNu_2J'

'DYreweight': ['data', 'DYJetsToLL_M-50', 'DYJetsToLL_0J', 'DYJetsToLL_1J', 'DYJetsToLL_2J', 'WJetsToLNu_TuneCP5', 'WJetsToLNu_0J', 'WJetsToLNu_1J', 'WJetsToLNu_2J', 'ZZ_TuneCP5', 'WZ_TuneCP5', 'WW_TuneCP5', 'TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu'],

'DYcorrected': ['data', 'DYJetsToLL_M-50', 'DYJetsToLL_0J', 'DYJetsToLL_1J', 'DYJetsToLL_2J', 'WJetsToLNu_TuneCP5', 'WJetsToLNu_0J', 'WJetsToLNu_1J', 'WJetsToLNu_2J', 'ZZ_TuneCP5', 'WZ_TuneCP5', 'WW_TuneCP5', 'TTToSemiLeptonic', 'TTToHadronic', 'TTTo2L2Nu']

}
