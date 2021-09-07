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

class MyDF(processor.ProcessorABC):
    def __init__(self, lumiWeight, BDTmodels, year):
        self._lumiWeight = lumiWeight
        self._BDTmodels = BDTmodels
        self._year = year
        self.var_0jet_ = ['e_met_mT_Per_e_m_Mass', 'm_met_mT_Per_e_m_Mass', 'mpt_Per_e_m_Mass', 'ept_Per_e_m_Mass', 'empt', 'met', 'DeltaR_e_m', 'emEta']
        self.var_1jet_ = ['e_met_mT_Per_e_m_Mass', 'm_met_mT_Per_e_m_Mass', 'mpt_Per_e_m_Mass', 'ept_Per_e_m_Mass', 'empt', 'met', 'DeltaR_e_m', 'emEta', 'j1pt', 'DeltaR_j1_em', 'j1Eta']
        self.var_2jet_ = ['mpt_Per_e_m_Mass', 'ept_Per_e_m_Mass', 'empt', 'met', 'DeltaR_e_m', 'emEta', 'j1pt', 'j1Eta', 'Rpt', 'j2pt', 'j2Eta', 'DeltaEta_j1_j2', 'pt_cen_Deltapt', 'j1_j2_mass', 'DeltaR_em_j1j2', 'Zeppenfeld_DeltaEta', 'DeltaPhi_j1_j2', 'DeltaR_j1_j2']
        dataset_axis = hist.Cat("dataset", "samples")
        self._accumulator = processor.dict_accumulator({
            'MVA_0jet': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("MVA_0jet", r"BDT Discriminator", 100, 0, 1),
            ),
            'MVA_1jet': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("MVA_1jet", r"BDT Discriminator", 100, 0, 1),
            ),
            'MVA_2jet_GG': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("MVA_2jet_GG", r"BDT Discriminator", 100, 0, 1),
            ),
            'MVA_2jet_VBF': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("MVA_2jet_VBF", r"BDT Discriminator", 100, 0, 1),
            )
        })
    @property
    def accumulator(self):
        return self._accumulator
    
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
        E_charge = ak.fill_none(ak.pad_none(E_collections.charge, 1), 0)
        M_charge = ak.fill_none(ak.pad_none(M_collections.charge, 1), 0)
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

        #Take the first leptons
        Electron_collections = Electron_collections[:,0]
        Muon_collections = Muon_collections[:,0]
        emVar = Electron_collections + Muon_collections

        if 'LFV' in emevents.metadata["dataset"]:
            massRange = (emVar.mass<160) & (emVar.mass>110)
        else:
            massRange = ((emVar.mass<115) & (emVar.mass>110)) | ((emVar.mass<160) & (emVar.mass>135))
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	
    
    def SF(self, emevents):
        if emevents.metadata["dataset"]=='SingleMuon' or emevents.metadata["dataset"] == 'data': 
           SF = ak.sum(emevents.Jet.passDeepJet_M,1)==0 #numpy.ones(len(emevents))
        else:
           #Get bTag SF
           #bTagSF_L = ak.prod(1-emevents.Jet.btagSF_deepjet_L*emevents.Jet.passDeepJet_L, axis=1)
           bTagSF_M = ak.prod(1-emevents.Jet.btagSF_deepjet_M*emevents.Jet.passDeepJet_M, axis=1)

           #PU/PF/Gen Weights
           if self._year == '2018':
             SF = emevents.puWeight*emevents.genWeight
           else:
             SF = emevents.puWeight*emevents.PrefireWeight*emevents.genWeight

           Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
           Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
           
           #Muon SF
           SF = bTagSF_M*SF*Muon_collections.Trigger_SF*Muon_collections.ID_SF*Muon_collections.ISO_SF

           #Electron SF and lumi
           SF = SF*Electron_collections.Reco_SF*Electron_collections.ID_SF*self._lumiWeight[emevents.metadata["dataset"]]

           SF = SF.to_numpy()
           SF[abs(SF)>10] = 0
        
        emevents["weight"] = SF
        emevents["label"] = numpy.ones(len(emevents), dtype=bool) if 'LFV' in emevents.metadata["dataset"] else numpy.zeros(len(emevents), dtype=bool)
        
        return emevents

    def BDTscore(self, njets, XFrame, isVBF=False):
        if isVBF:
           model_load = self._BDTmodels["model_VBF_2jets"]
        else:
           model_load = self._BDTmodels[f"model_GG_{njets}jets"]
        return model_load.predict_proba(XFrame)

    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        #make interesting variables
        #zero/any no. of jets
        emVar = Electron_collections + Muon_collections
        emevents["eEta"] = Electron_collections.eta
        emevents["mEta"] = Muon_collections.eta
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

        #one jets
        onejets_emevents = emevents[emevents.nJet30 >= 1]
        Electron_collections_1jet = Electron_collections[emevents.nJet30 >= 1]
        Muon_collections_1jet = Muon_collections[emevents.nJet30 >= 1]
        emVar_1jet = Electron_collections_1jet + Muon_collections_1jet
        Jet_collections_1jet = Jet_collections[emevents.nJet30 >= 1]

        onejets_emevents['j1pt'] = Jet_collections_1jet[:,0].pt
        onejets_emevents['j1Eta'] = Jet_collections_1jet[:,0].eta

        onejets_emevents["DeltaEta_j1_em"] = abs(Jet_collections_1jet[:,0].eta - emVar_1jet.eta)
        onejets_emevents["DeltaPhi_j1_em"] = Jet_collections_1jet[:,0].delta_phi(emVar_1jet)
        onejets_emevents["DeltaR_j1_em"] = Jet_collections_1jet[:,0].delta_r(emVar_1jet)

        onejets_emevents["Zeppenfeld_1"] = Zeppenfeld(Muon_collections_1jet, Electron_collections_1jet, [Jet_collections_1jet[:,0]])
        onejets_emevents["Rpt_1"] = Rpt(Muon_collections_1jet, Electron_collections_1jet, [Jet_collections_1jet[:,0]])

        #2 or more jets
        Multijets_emevents = onejets_emevents[onejets_emevents.nJet30 >= 2]
        Electron_collections_2jet = Electron_collections_1jet[onejets_emevents.nJet30 >= 2]
        Muon_collections_2jet = Muon_collections_1jet[onejets_emevents.nJet30 >= 2]
        emVar_2jet = Electron_collections_2jet + Muon_collections_2jet
        Jet_collections_2jet = Jet_collections_1jet[onejets_emevents.nJet30 >= 2]

        Multijets_emevents['j2pt'] = Jet_collections_2jet[:,1].pt
        Multijets_emevents['j2Eta'] = Jet_collections_2jet[:,1].eta
        Multijets_emevents["j1_j2_mass"] = (Jet_collections_2jet[:,0] + Jet_collections_2jet[:,1]).mass

        Multijets_emevents["DeltaEta_em_j1j2"] = abs((Jet_collections_2jet[:,0] + Jet_collections_2jet[:,1]).eta - emVar_2jet.eta)
        Multijets_emevents["DeltaPhi_em_j1j2"] = (Jet_collections_2jet[:,0] + Jet_collections_2jet[:,1]).delta_phi(emVar_2jet)
        Multijets_emevents["DeltaR_em_j1j2"] = (Jet_collections_2jet[:,0] + Jet_collections_2jet[:,1]).delta_r(emVar_2jet)

        Multijets_emevents["DeltaEta_j2_em"] = abs(Jet_collections_2jet[:,1].eta - emVar_2jet.eta)
        Multijets_emevents["DeltaPhi_j2_em"] = Jet_collections_2jet[:,1].delta_phi(emVar_2jet)
        Multijets_emevents["DeltaR_j2_em"] = Jet_collections_2jet[:,1].delta_r(emVar_2jet)

        Multijets_emevents["DeltaEta_j1_j2"] = abs(Jet_collections_2jet[:,0].eta - Jet_collections_2jet[:,1].eta)

        Multijets_emevents["isVBFcat"] = ((Multijets_emevents["j1_j2_mass"] > 400) & (Multijets_emevents["DeltaEta_j1_j2"] > 2.5)) 

        Multijets_emevents["DeltaPhi_j1_j2"] = Jet_collections_2jet[:,0].delta_phi(Jet_collections_2jet[:,1])
        Multijets_emevents["DeltaR_j1_j2"] = Jet_collections_2jet[:,0].delta_r(Jet_collections_2jet[:,1])

        Multijets_emevents["Zeppenfeld"] = Zeppenfeld(Muon_collections_2jet, Electron_collections_2jet, [Jet_collections_2jet[:,0], Jet_collections_2jet[:,1]])
        Multijets_emevents["Zeppenfeld_DeltaEta"] = Multijets_emevents["Zeppenfeld"]/Multijets_emevents["DeltaEta_j1_j2"]
        Multijets_emevents["absZeppenfeld_DeltaEta"] = abs(Multijets_emevents["Zeppenfeld_DeltaEta"])
        Multijets_emevents["cen"] = numpy.exp(-4*Multijets_emevents["Zeppenfeld_DeltaEta"]**2)

        Multijets_emevents["Rpt"] = Rpt(Muon_collections_2jet, Electron_collections_2jet, [Jet_collections_2jet[:,0], Jet_collections_2jet[:,1]])

        Multijets_emevents["pt_cen"] = pt_cen(Muon_collections_2jet, Electron_collections_2jet, [Jet_collections_2jet[:,0], Jet_collections_2jet[:,1]])
        Multijets_emevents["pt_cen_Deltapt"] = Multijets_emevents["pt_cen"]/(Jet_collections_2jet[:,0] - Jet_collections_2jet[:,1]).pt
        Multijets_emevents["abspt_cen_Deltapt"] = abs(Multijets_emevents["pt_cen_Deltapt"])

        Multijets_emevents["Ht_had"] = ak.sum(Jet_collections_2jet.pt, 1)
        Multijets_emevents["Ht"] = ak.sum(Jet_collections_2jet.pt, 1) + Muon_collections_2jet.pt + Electron_collections_2jet.pt
        return emevents, onejets_emevents, Multijets_emevents[Multijets_emevents.isVBFcat==0], Multijets_emevents[Multijets_emevents.isVBFcat==1]

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()

        emevents = self.Vetos(events)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = self.Corrections(emevents)
          emevents = self.SF(emevents)
          emevents, onejets_emevents, Multijets_emevents_GG, Multijets_emevents_VBF = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)

          sample_group_name = "" 
          if "ST" in emevents.metadata["dataset"] or "TT" in emevents.metadata["dataset"]:
            sample_group_name = r'$t\bar{t}$,t+Jets'
          elif "HTo" in emevents.metadata["dataset"] and not "LFV" in emevents.metadata["dataset"]:
            sample_group_name = 'SM Higgs'
          elif "ZZ" in emevents.metadata["dataset"] or "WZ" in emevents.metadata["dataset"] or "WW" in emevents.metadata["dataset"]:
            sample_group_name = "Diboson"
          elif "DY" in emevents.metadata["dataset"]:
            sample_group_name = "DY+Jets"
          elif "JetsToLNu" in emevents.metadata["dataset"] or "WG" in emevents.metadata["dataset"]:
            sample_group_name = "W+Jets"
          elif "EWK" in emevents.metadata["dataset"]:
            sample_group_name = "EWK W/Z"
          elif "data" in emevents.metadata["dataset"]:
            sample_group_name = "data"
          elif "LFV" in emevents.metadata["dataset"]:
            sample_group_name = r'$H\rightarrow e\mu$ (BR=1%)'

          Xframe_0jet = ak.to_pandas(emevents[emevents.nJet30 == 0][self.var_0jet_])
          Xframe_1jet = ak.to_pandas(onejets_emevents[onejets_emevents.nJet30 == 1][self.var_1jet_])
          Xframe_2jet_GG = ak.to_pandas(Multijets_emevents_GG[self.var_2jet_])
          Xframe_2jet_VBF = ak.to_pandas(Multijets_emevents_VBF[self.var_2jet_])

          out['MVA_0jet'].fill(
              dataset=sample_group_name,
              MVA_0jet=self.BDTscore(0, Xframe_0jet)[:,1], 
              weight=emevents[emevents.nJet30 == 0]["weight"]
          )
          out['MVA_1jet'].fill(
              dataset=sample_group_name,
              MVA_1jet=self.BDTscore(1, Xframe_1jet)[:,1], 
              weight=onejets_emevents[onejets_emevents.nJet30 == 1]["weight"]
          )
          out['MVA_2jet_GG'].fill(
              dataset=sample_group_name,
              MVA_2jet_GG=self.BDTscore(2, Xframe_2jet_GG)[:,1], 
              weight=Multijets_emevents_GG["weight"]
          )
          out['MVA_2jet_VBF'].fill(
              dataset=sample_group_name,
              MVA_2jet_VBF=self.BDTscore(2, Xframe_2jet_VBF)[:,1], 
              weight=Multijets_emevents_VBF["weight"]
          )

        return out

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
  BDTjsons = ['model_GG_0jets', 'model_GG_1jets', 'model_GG_2jets', 'model_VBF_2jets']
  BDTmodels = {}
  for BDTjson in BDTjsons:
    BDTmodels[BDTjson] = xgb.XGBClassifier()
    BDTmodels[BDTjson].load_model(f'XGBoost-for-HtoEMu/models/{BDTjson}.bin')
  print(BDTmodels)
  years = ['2017', '2018']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
    processor_instance = MyDF(lumiWeight, BDTmodels, year)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
