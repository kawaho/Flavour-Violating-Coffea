import numpy
import awkward as ak
class BDT_functions:
    def __init__(self, _BDTmodels, var_GG_, var_2jet_VBF_):
        self._BDTmodels = _BDTmodels
        self.var_GG_ = var_GG_
        self.var_2jet_VBF_ = var_2jet_VBF_
    
    def BDTscore(self, XFrame, isVBF=False):
        if isVBF:
           model_load = self._BDTmodels["model_vbf_v9"]
        else:
           model_load = self._BDTmodels[f"model_gg_v9"]
        return model_load.predict_proba(XFrame)

    def pandasDF(self, emevents, unc=None, UpDown=None, isJetSys=False):
        if unc==None:
#          for i in self.var_GG_:
#            print(i, emevents[i].type)
#            for k in emevents[i]: print(k)
          Xframe_GG = ak.to_pandas(emevents[self.var_GG_]).fillna(value=numpy.nan)
          Xframe_2jet_VBF = ak.to_pandas(emevents[self.var_2jet_VBF_])
 
          emevents_GG_nom = self.BDTscore(Xframe_GG)[:,1] 
          emevents_2jet_VBF_nom = self.BDTscore(Xframe_2jet_VBF, True)[:,1] 
          emevents['mva'] = (emevents.isVBFcat==0) * emevents_GG_nom + ((emevents.njets>=2) & (emevents.isVBFcat==1))* emevents_2jet_VBF_nom
        else:
          var_GG_alt = [i+f'_{unc}_{UpDown}' if i+f'_{unc}_{UpDown}' in emevents.fields else i for i in self.var_GG_] 
          var_2jet_VBF_alt = [i+f'_{unc}_{UpDown}' if i+f'_{unc}_{UpDown}' in emevents.fields else i for i in self.var_2jet_VBF_] 

          Xframe_GG = ak.to_pandas(emevents[var_GG_alt]).fillna(value=numpy.nan)
          Xframe_2jet_VBF = ak.to_pandas(emevents[var_2jet_VBF_alt])
  
          rename2_GG_ = {i:i.replace(f'_{unc}_{UpDown}', '') for i in var_GG_alt}
          rename2_VBF_ = {i:i.replace(f'_{unc}_{UpDown}', '') for i in var_2jet_VBF_alt}

          Xframe_GG.rename(columns=rename2_GG_, inplace = True)
          Xframe_2jet_VBF.rename(columns=rename2_VBF_, inplace = True)
     
          emevents_GG = self.BDTscore(Xframe_GG)[:,1] 
          emevents_2jet_VBF = self.BDTscore(Xframe_2jet_VBF, True)[:,1] 
          if isJetSys:
            emevents[f'mva_{unc}_{UpDown}'] = (emevents[f"isVBFcat_{unc}_{UpDown}"]==0) * emevents_GG + ((emevents[f"njets_{unc}_{UpDown}"]>=2) & (emevents[f"isVBFcat_{unc}_{UpDown}"]==1))* emevents_2jet_VBF
          else:
            emevents[f'mva_{unc}_{UpDown}'] = (emevents.isVBFcat==0) * emevents_GG + ((emevents.njets>=2) & (emevents.isVBFcat==1))* emevents_2jet_VBF
        return emevents
