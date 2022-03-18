from coffea import processor, hist
from coffea.util import save
import xgboost as xgb
import awkward as ak
import numpy, json, os

def pZeta(leg1, leg2, MET_px, MET_py):
    leg1x = numpy.cos(leg1.phi)
    leg2x = numpy.cos(leg2.phi)
    leg1y = numpy.sin(leg1.phi)
    leg2y = numpy.sin(leg2.phi)
    zetaX = leg1x + leg2x
    zetaY = leg1y + leg2y
    zetaR = numpy.sqrt(zetaX*zetaX + zetaY*zetaY)
    
    zetaX = numpy.where((zetaR > 0.), zetaX/zetaR, zetaX)
    zetaY = numpy.where((zetaR > 0.), zetaY/zetaR, zetaY)
    
    visPx = leg1.px + leg2.px
    visPy = leg1.py + leg2.py
    pZetaVis = visPx*zetaX + visPy*zetaY
    px = visPx + MET_px
    py = visPy + MET_py
    
    pZeta = px*zetaX + py*zetaY
    
    return (pZeta, pZetaVis)

def Rpt(lep1, lep2, jets=None):
    emVar = lep1+lep2
    if jets==None:
        return (emVar).pt/(lep1.pt+lep2.pt)
    elif len(jets)==1:
        return (emVar + jets[0]).pt/(lep1.pt+lep2.pt+jets[0].pt)
    elif len(jets)==2:
        return (emVar + jets[0] +jets[1]).pt/(lep1.pt+lep2.pt+jets[0].pt+jets[1].pt)
    else:
        return -999
    
def Zeppenfeld(lep1, lep2, jets):
    emVar = lep1+lep2
    if len(jets)==1:
        return emVar.eta - (jets[0].eta)/2
    elif len(jets)==2:
        return emVar.eta - (jets[0].eta + jets[1].eta)/2
    else:
        return -999

def mT(lep, met):
    return numpy.sqrt(abs((numpy.sqrt(lep.mass**2+lep.pt**2) + met.pt)**2 - (lep+met).pt**2))

def pt_cen(lep1, lep2, jets):
    emVar = lep1+lep2
    if len(jets)==1:
        return emVar.pt - jets[0].pt/2
    elif len(jets)==2:
        return emVar.pt - (jets[0] + jets[1]).pt/2
    else:
        return -999

class MyEMuPeak(processor.ProcessorABC):
    def __init__(self, lumiWeight, BDTmodels, BDTvars, year):
        self._lumiWeight = lumiWeight
        self._BDTmodels = BDTmodels
        self._year = year
        self.var_GG_ = BDTvars['model_GG']
        self.var_2jet_VBF_ = BDTvars['model_VBF']
        self._accumulator = processor.dict_accumulator({})
        self._accumulator[f'e_m_Mass'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'mva'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'isVBFcat'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'njets'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'weight'] = processor.column_accumulator(numpy.array([]))
  
    @property
    def accumulator(self):
        return self._accumulator
    
    def BDTscore(self, XFrame, isVBF=False):
        if isVBF:
           model_load = self._BDTmodels["model_VBF"]
        else:
           model_load = self._BDTmodels[f"model_GG"]
        return model_load.predict_proba(XFrame)

    def Vetos(self, events):
        if self._year == '2016preVFP':
          mpt_threshold = 26
          trigger = events.HLT.IsoMu24 | events.HLT.IsoTkMu24
        elif self._year == '2016postVFP':
          mpt_threshold = 26
          trigger = events.HLT.IsoMu24 | events.HLT.IsoTkMu24
        elif self._year == '2017':
          mpt_threshold = 29
          trigger = events.HLT.IsoMu27
        elif self._year == '2018':
          mpt_threshold = 26
          trigger = events.HLT.IsoMu24

        #Choose em channel and IsoMu Trigger
        emevents = events[(events.channel == 0) & (trigger == 1)]

        E_collections = emevents.Electron
        M_collections = emevents.Muon

        #Kinematics Selections
        emevents["Electron", "Target"] = ((E_collections.pt > 24) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.05) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
        emevents["Muon", "Target"] = ((M_collections.pt > mpt_threshold) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.05) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

        E_collections = emevents.Electron[emevents.Electron.Target==1]
        M_collections = emevents.Muon[emevents.Muon.Target==1]

        #Opposite Charge
        E_charge = ak.fill_none(ak.pad_none(E_collections.charge, 1, axis=-1), 0, axis=-1)[:,0]
        M_charge = ak.fill_none(ak.pad_none(M_collections.charge, 1, axis=-1), 0, axis=-1)[:,0]
        opp_charge = E_charge*M_charge==-1
        emevents = emevents[opp_charge]

        #Trig Matching
        M_collections = emevents.Muon
        trg_collections = emevents.TrigObj

        M_collections = M_collections[M_collections.Target==1]
        #https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L60
        trg_collections = trg_collections[trg_collections.id == 13]

        trg_Match = ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1)
        return emevents[trg_Match]
   
    def SF(self, emevents):
        emevents['weight'] = ak.sum(emevents.Jet.passDeepJet_L,1)==0 
        return emevents

    def Corrections(self, emevents):
        Electron_collections = emevents.Electron[emevents.Electron.Target==1]
        Muon_collections = emevents.Muon[emevents.Muon.Target==1]
        MET_collections = emevents.MET
        Jet_collections = emevents.Jet[emevents.Jet.passJet30ID==1]

        #Jet corrections
        Jet_collections['pt'] = Jet_collections['pt_nom']
        Jet_collections['mass'] = Jet_collections['mass_nom']

        #MET corrections
        MET_collections['phi'] = MET_collections['T1_phi'] 
        MET_collections['pt'] = MET_collections['T1_pt'] 

        #Muon pT corrections
        Muon_collections['pt'] = Muon_collections['corrected_pt']

        #ensure Jets are pT-ordered
        Jet_collections = Jet_collections[ak.argsort(Jet_collections.pt, axis=1, ascending=False)]
        Jet_collections = ak.pad_none(Jet_collections, 2)

        #Take the first leptons
        Electron_collections = Electron_collections[:,0]
        Muon_collections = Muon_collections[:,0]
        emVar = Electron_collections + Muon_collections

        massRange = (emVar.mass<180) & (emVar.mass>90)
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	
    
    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        emVar = Electron_collections + Muon_collections
        emevents["e_m_Mass"] = emVar.mass
        #emevents["mpt_Per_e_m_Mass"] = Muon_collections.pt/emVar.mass
        #emevents["ept_Per_e_m_Mass"] = Electron_collections.pt/emVar.mass
        emevents["empt"] = emVar.pt
        emevents["DeltaEta_e_m"] = abs(Muon_collections.eta - Electron_collections.eta)
        emevents["met"] = MET_collections.pt
        emevents["DeltaPhi_em_met"] = emVar.delta_phi(MET_collections)
        #emevents["e_met_mT"] = mT(Electron_collections, MET_collections)
        #emevents["m_met_mT"] = mT(Muon_collections, MET_collections)
        #pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
        #emevents["pZeta85"] = pZeta_ - 0.85*pZetaVis_
        emevents["njets"] = emevents.nJet30 
        #1 jet
        emevents['j1pt'] = Jet_collections[:,0].pt
        emevents['j1Eta'] = Jet_collections[:,0].eta
        emevents["DeltaEta_j1_em"] = abs(Jet_collections[:,0].eta - emVar.eta)

        #2 or more jets
        emevents['j2pt'] = Jet_collections[:,1].pt
        emevents["j1_j2_mass"] = (Jet_collections[:,0] + Jet_collections[:,1]).mass
        emevents["DeltaEta_em_j1j2"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - emVar.eta)
        emevents["DeltaEta_j1_j2"] = abs(Jet_collections[:,0].eta - Jet_collections[:,1].eta)
        emevents["isVBFcat"] = ((emevents["njets"] >= 2) & (emevents["j1_j2_mass"] > 400) & (emevents["DeltaEta_j1_j2"] > 2.5)) 
        emevents["isVBFcat"] = ak.fill_none(emevents["isVBFcat"], 0)
        emevents["Zeppenfeld_DeltaEta"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/emevents["DeltaEta_j1_j2"]
        emevents["Rpt"] = Rpt(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
        emevents["pt_cen_Deltapt"] = pt_cen(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/(Jet_collections[:,0] - Jet_collections[:,1]).pt
        emevents["Ht_had"] = ak.sum(Jet_collections.pt, 1)

        return emevents

    def pandasDF(self, emevents):
        Xframe_GG = ak.to_pandas(emevents[self.var_GG_]).fillna(value=numpy.nan)
        Xframe_2jet_VBF = ak.to_pandas(emevents[self.var_2jet_VBF_])
        emevents_GG_nom = self.BDTscore(Xframe_GG)[:,1] 
        emevents_2jet_VBF_nom = self.BDTscore(Xframe_2jet_VBF, True)[:,1] 
        emevents['mva'] = (emevents.isVBFcat==0) * emevents_GG_nom + ((emevents.njets>=2) & (emevents.isVBFcat==1))* emevents_2jet_VBF_nom
        return emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        if len(emevents)==0: return out
        emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = self.Corrections(emevents)
        if len(emevents)==0: return out
        emevents = self.SF(emevents)
        emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
        emevents = self.pandasDF(emevents)
        for sys_var_ in out:
          acc = emevents[sys_var_].to_numpy()
          out[sys_var_].add( processor.column_accumulator( acc ) )
        #else:
        #  print("No Events found in "+emevents.metadata["dataset"]) 
        return out

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
  BDTjsons = ['model_GG', 'model_VBF']
  BDTmodels = {}
  BDTvars = {}
  for BDTjson in BDTjsons:
    BDTmodels[BDTjson] = xgb.XGBClassifier()
    BDTmodels[BDTjson].load_model(f'XGBoost-for-HtoEMu/results/{BDTjson}.json')
    BDTvars[BDTjson] = BDTmodels[BDTjson].get_booster().feature_names

    print(BDTmodels[BDTjson].get_booster().feature_names)
  print(BDTmodels)
  years = ['2016preVFP', '2016postVFP', '2017', '2018']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
    processor_instance = MyEMuPeak(lumiWeight, BDTmodels, BDTvars, year)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
