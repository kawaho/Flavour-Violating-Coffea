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
    def __init__(self, lumiWeight, BDTmodels, year):
        self._lumiWeight = lumiWeight
        self._BDTmodels = BDTmodels
        self._year = year
        self.var_0jet_ = ['e_met_mT', 'm_met_mT', 'mpt_Per_e_m_Mass', 'ept_Per_e_m_Mass', 'empt', 'met', 'DeltaEta_e_m', 'emEta', 'pZeta15'] 
        self.var_1jet_ = self.var_0jet_ #['e_met_mT_Per_e_m_Mass', 'm_met_mT_Per_e_m_Mass', 'mpt_Per_e_m_Mass', 'ept_Per_e_m_Mass', 'empt', 'met', 'DeltaR_e_m', 'emEta', 'j1pt', 'DeltaR_j1_em', 'j1Eta']
        self.var_2jet_GG_ = self.var_0jet_ #['mpt_Per_e_m_Mass', 'ept_Per_e_m_Mass', 'empt', 'met', 'DeltaR_e_m', 'emEta', 'j1pt', 'j1Eta', 'Rpt', 'j2pt', 'j2Eta', 'DeltaEta_j1_j2', 'pt_cen_Deltapt', 'j1_j2_mass', 'DeltaR_em_j1j2', 'Zeppenfeld_DeltaEta', 'DeltaPhi_j1_j2', 'DeltaR_j1_j2']
        self.var_2jet_VBF_ = self.var_0jet_ #['mpt_Per_e_m_Mass', 'ept_Per_e_m_Mass', 'empt', 'met', 'DeltaR_e_m', 'emEta', 'j1pt', 'j1Eta', 'Rpt', 'j2pt', 'j2Eta', 'DeltaEta_j1_j2', 'pt_cen_Deltapt', 'j1_j2_mass', 'DeltaR_em_j1j2', 'Zeppenfeld_DeltaEta', 'DeltaPhi_j1_j2', 'DeltaR_j1_j2']
        self._accumulator = processor.dict_accumulator({})
        for cat in ['GG0', 'GG1', 'GG2', 'VBF2']:
          self._accumulator[f'e_m_Mass_{cat}'] = processor.column_accumulator(numpy.array([]))
          self._accumulator[f'mva_{cat}'] = processor.column_accumulator(numpy.array([]))
          self._accumulator[f'weight_{cat}'] = processor.column_accumulator(numpy.array([]))
  
    @property
    def accumulator(self):
        return self._accumulator
    
    def BDTscore(self, njets, XFrame, isVBF=False):
        if isVBF:
           model_load = self._BDTmodels["model_GG_0jets"]#model_VBF_2jets"]
        else:
           model_load = self._BDTmodels[f"model_GG_0jets"]#model_GG_{njets}jets"]
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
        emevents["Electron", "Target"] = ((E_collections.pt > 24) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.045) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
        emevents["Muon", "Target"] = ((M_collections.pt > mpt_threshold) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.045) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

        E_collections = emevents.Electron[emevents.Electron.Target==1]
        M_collections = emevents.Muon[emevents.Muon.Target==1]

        #Opposite Charge
        E_charge = ak.fill_none(ak.pad_none(E_collections.charge, 1, axis=-1), 0, axis=-1)
        M_charge = ak.fill_none(ak.pad_none(M_collections.charge, 1, axis=-1), 0, axis=-1)
        opp_charge = ak.flatten(E_charge*M_charge==-1)

        emevents = emevents[opp_charge]

        #Trig Matching
        M_collections = emevents.Muon
        trg_collections = emevents.TrigObj

        M_collections = M_collections[M_collections.Target==1]
        #https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L60
        trg_collections = trg_collections[trg_collections.id == 13]

        trg_Match = ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1)

        return emevents[trg_Match]
   
    def Corrections(self, emevents):
        Electron_collections = emevents.Electron[emevents.Electron.Target==1]
        Muon_collections = emevents.Muon[emevents.Muon.Target==1]
        MET_collections = emevents.MET
        Jet_collections = emevents.Jet[emevents.Jet.passJet30ID==1]

        #Jet corrections
        Jet_collections['pt'] = Jet_collections['pt_nom']
        Jet_collections['mass'] = Jet_collections['mass_nom']

        #MET corrections
        if emevents.metadata["dataset"]!='SingleMuon' and emevents.metadata["dataset"]!='data':
            MET_collections['phi'] = MET_collections['T1Smear_phi'] 
            MET_collections['pt'] = MET_collections['T1Smear_pt'] 
        else:
            MET_collections['phi'] = MET_collections['T1_phi'] 
            MET_collections['pt'] = MET_collections['T1_pt'] 

        #MET corrections Electron
        Electron_collections['pt'] = Electron_collections['pt']/Electron_collections['eCorr']
        MET_collections = MET_collections+Electron_collections[:,0]
        Electron_collections['pt'] = Electron_collections['pt']*Electron_collections['eCorr']
        MET_collections = MET_collections-Electron_collections[:,0]
        
        #Muon pT corrections
        MET_collections = MET_collections+Muon_collections[:,0]
        Muon_collections['mass'] = Muon_collections['mass']*Muon_collections['corrected_pt']/Muon_collections['pt']
        Muon_collections['pt'] = Muon_collections['corrected_pt']
        MET_collections = MET_collections-Muon_collections[:,0]

        #ensure Jets are pT-ordered
        Jet_collections = Jet_collections[ak.argsort(Jet_collections.pt, axis=1, ascending=False)]
        #padding to have at least "2 jets"
        Jet_collections = ak.pad_none(Jet_collections, 2, clip=True)

        #Take the first leptons
        Electron_collections = Electron_collections[:,0]
        Muon_collections = Muon_collections[:,0]
        emVar = Electron_collections + Muon_collections

        massRange = (emVar.mass<160) & (emVar.mass>110)
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	
    
    def SF(self, emevents):
        passDeepJet30 = (emevents.Jet.pt_nom > 30) & emevents.Jet.passDeepJet_L
        SF = ak.sum(passDeepJet30,1)==0 #numpy.ones(len(emevents))
        emevents["weight"] = SF
        return emevents 

    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        #make interesting variables
        #zero/any no. of jets
        emVar = Electron_collections + Muon_collections
        emevents["eEta"] = Electron_collections.eta
        emevents["mEta"] = Muon_collections.eta

        emevents["e_m_Mass"] = emVar.mass
        emevents["mpt_Per_e_m_Mass"] = Muon_collections.pt/emVar.mass
        emevents["ept_Per_e_m_Mass"] = Electron_collections.pt/emVar.mass
        emevents["empt"] = emVar.pt
        emevents["emEta"] = emVar.eta
        emevents["DeltaEta_e_m"] = abs(Muon_collections.eta - Electron_collections.eta)
        emevents["DeltaPhi_e_m"] = Muon_collections.delta_phi(Electron_collections)
        emevents["DeltaR_e_m"] = Muon_collections.delta_r(Electron_collections)
        emevents["Rpt_0"] = Rpt(Muon_collections, Electron_collections)

        emevents["met"] = MET_collections.pt

        emevents["e_met_mT"] = mT(Electron_collections, MET_collections)
        emevents["m_met_mT"] = mT(Muon_collections, MET_collections)
        emevents["e_met_mT_Per_e_m_Mass"] = emevents["e_met_mT"]/emVar.mass
        emevents["m_met_mT_Per_e_m_Mass"] = emevents["m_met_mT"]/emVar.mass

        pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
        emevents["pZeta85"] = pZeta_ - 0.85*pZetaVis_
        emevents["pZeta15"] = pZeta_ - 1.5*pZetaVis_
        emevents["pZeta"] = pZeta_
        emevents["pZetaVis"] = pZetaVis_

        #1 jet
        emevents['j1pt'] = Jet_collections[:,0].pt
        emevents['j1Eta'] = Jet_collections[:,0].eta

        emevents["DeltaEta_j1_em"] = abs(Jet_collections[:,0].eta - emVar.eta)
        emevents["DeltaPhi_j1_em"] = Jet_collections[:,0].delta_phi(emVar)
        emevents["DeltaR_j1_em"] = Jet_collections[:,0].delta_r(emVar)

        emevents["Zeppenfeld_1"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0]])
        emevents["Rpt_1"] = Rpt(Muon_collections, Electron_collections, [Jet_collections[:,0]])

        #2 or more jets
        emevents['j2pt'] = Jet_collections[:,1].pt
        emevents['j2Eta'] = Jet_collections[:,1].eta
        emevents["j1_j2_mass"] = (Jet_collections[:,0] + Jet_collections[:,1]).mass

        emevents["DeltaEta_em_j1j2"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - emVar.eta)
        emevents["DeltaPhi_em_j1j2"] = (Jet_collections[:,0] + Jet_collections[:,1]).delta_phi(emVar)
        emevents["DeltaR_em_j1j2"] = (Jet_collections[:,0] + Jet_collections[:,1]).delta_r(emVar)

        emevents["DeltaEta_j2_em"] = abs(Jet_collections[:,1].eta - emVar.eta)
        emevents["DeltaPhi_j2_em"] = Jet_collections[:,1].delta_phi(emVar)
        emevents["DeltaR_j2_em"] = Jet_collections[:,1].delta_r(emVar)

        emevents["DeltaEta_j1_j2"] = abs(Jet_collections[:,0].eta - Jet_collections[:,1].eta)

        emevents["isVBFcat"] = ((emevents["nJet30"] == 2) & (emevents["j1_j2_mass"] > 400) & (emevents["DeltaEta_j1_j2"] > 2.5)) 
        emevents["isVBFcat"] = ak.fill_none(emevents["isVBFcat"], 0)

        emevents["DeltaPhi_j1_j2"] = Jet_collections[:,0].delta_phi(Jet_collections[:,1])
        emevents["DeltaR_j1_j2"] = Jet_collections[:,0].delta_r(Jet_collections[:,1])

        emevents["Zeppenfeld"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
        emevents["Zeppenfeld_DeltaEta"] = emevents["Zeppenfeld"]/emevents["DeltaEta_j1_j2"]
        emevents["absZeppenfeld_DeltaEta"] = abs(emevents["Zeppenfeld_DeltaEta"])
        emevents["cen"] = numpy.exp(-4*emevents["Zeppenfeld_DeltaEta"]**2)

        emevents["Rpt"] = Rpt(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])

        emevents["pt_cen"] = pt_cen(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
        emevents["pt_cen_Deltapt"] = emevents["pt_cen"]/(Jet_collections[:,0] - Jet_collections[:,1]).pt
        emevents["abspt_cen_Deltapt"] = abs(emevents["pt_cen_Deltapt"])

        emevents["Ht_had"] = ak.sum(Jet_collections.pt, 1)
        emevents["Ht"] = ak.sum(Jet_collections.pt, 1) + Muon_collections.pt + Electron_collections.pt

        return emevents

    def pandasDF(self, emevents_0jet, emevents_1jet, emevents_2jet_GG, emevents_2jet_VBF):
        Xframe_0jet = ak.to_pandas(emevents_0jet[self.var_0jet_])
        Xframe_1jet = ak.to_pandas(emevents_1jet[self.var_1jet_])
        Xframe_2jet_GG = ak.to_pandas(emevents_2jet_GG[self.var_2jet_GG_])
        Xframe_2jet_VBF = ak.to_pandas(emevents_2jet_VBF[self.var_2jet_VBF_])
   
        if len(Xframe_0jet.index)==0:
          pass
        else:
          emevents_0jet['mva'] = self.BDTscore(0, Xframe_0jet)[:,1] 
        if len(Xframe_1jet.index)==0:
          pass
        else:
          emevents_1jet['mva'] = self.BDTscore(1, Xframe_1jet)[:,1] 
        if len(Xframe_2jet_GG.index)==0:
          pass
        else:
          emevents_2jet_GG['mva'] = self.BDTscore(2, Xframe_2jet_GG)[:,1] 
        if len(Xframe_2jet_VBF.index)==0:
          pass
        else:
          emevents_2jet_VBF['mva'] = self.BDTscore(2, Xframe_2jet_VBF)[:,1] 

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = self.Corrections(emevents)
          emevents = self.SF(emevents)
          emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
          emevents_0jet, emevents_1jet, emevents_2jet_GG, emevents_2jet_VBF = emevents[emevents.nJet30 == 0], emevents[emevents.nJet30 == 1], emevents[(emevents.nJet30==2) & (emevents.isVBFcat==0)], emevents[(emevents.nJet30==2) & (emevents.isVBFcat==1)]
          self.pandasDF(emevents_0jet, emevents_1jet, emevents_2jet_GG, emevents_2jet_VBF)

          for sys_var_ in out:
            if 'GG0' in sys_var_:
              if len(emevents_0jet)==0: continue
              acc = emevents_0jet[sys_var_.replace('_GG0','')].to_numpy()
            elif 'GG1' in sys_var_:
              if len(emevents_1jet)==0: continue
              acc = emevents_1jet[sys_var_.replace('_GG1','')].to_numpy()
            elif 'GG2' in sys_var_:
              if len(emevents_2jet_GG)==0: continue
              acc = emevents_2jet_GG[sys_var_.replace('_GG2','')].to_numpy()
            elif 'VBF2' in sys_var_:
              if len(emevents_2jet_VBF)==0: continue
              acc = emevents_2jet_VBF[sys_var_.replace('_VBF2','')].to_numpy()

            out[sys_var_].add( processor.column_accumulator( acc ) )
        else:
          print("No Events found in "+emevents.metadata["dataset"]) 
        return out

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
  BDTjsons = ['model_GG_0jets', 'model_GG_0jets', 'model_GG_0jets', 'model_GG_0jets'] #'model_GG_1jets', 'model_GG_2jets', 'model_VBF_2jets']
  BDTmodels = {}
  for BDTjson in BDTjsons:
    BDTmodels[BDTjson] = xgb.XGBClassifier()
    BDTmodels[BDTjson].load_model(f'{BDTjson}.bin')
  print(BDTmodels)
  years = ['2017', '2018']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
    processor_instance = MyEMuPeak(lumiWeight, BDTmodels, year)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
