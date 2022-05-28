from coffea import processor, hist
from coffea.util import save
from coffea.lookup_tools import extractor
from coffea.btag_tools import BTagScaleFactor
import xgboost as xgb
import pandas as pd
import awkward as ak
import correctionlib, numpy, json, os

class MyEMuPeak(processor.ProcessorABC):
    def __init__(self, lumiWeight, year, btag_sf, muon_sf, electron_sf, evaluator):
        self._lumiWeight = lumiWeight
        self._btag_sf = btag_sf
        self._e_sf = electron_sf
        self._m_sf = muon_sf
        self._evaluator = evaluator
        self._year = year
        self._accumulator = processor.dict_accumulator({})
        self._accumulator[f'hist'] = hist.Hist("Events", hist.Cat("sample", "sample name"), hist.Bin("emmass", "emmass", 450, 90, 160))

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
        emevents = events#[(events.channel == 0) & (trigger == 1)]

        E_collections = emevents.Electron
        M_collections = emevents.Muon

        #Kinematics Selections
        emevents["Electron", "Target"] = ((abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.05) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
        emevents["Muon", "Target"] = ((abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.05) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))
#        emevents["Electron", "Target"] = ((E_collections.pt > 24) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.05) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
#        emevents["Muon", "Target"] = ((M_collections.pt > mpt_threshold) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.05) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

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

        return emevents#[trg_Match]
   
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
        #padding to have at least "2 jets"
        Jet_collections = ak.pad_none(Jet_collections, 2)

        #Take the first leptons
        Electron_collections = Electron_collections[:,0]
        Muon_collections = Muon_collections[:,0]
        emVar = Electron_collections + Muon_collections

        massRange = (emVar.mass<160) & (emVar.mass>90)
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	
    
    def SF(self, emevents):
        #Get bTag SF
        #jet_flat = ak.flatten(emevents.Jet)
        #btagSF_deepjet_L = numpy.zeros(len(jet_flat))
        #jet_light = ak.where((jet_flat.passDeepJet_L) & (jet_flat.hadronFlavour==0))
        #jet_heavy = ak.where((jet_flat.passDeepJet_L) & (jet_flat.hadronFlavour!=0))
        #array_light = self._btag_sf["deepJet_incl"].evaluate("central", "L", jet_flat[jet_light].hadronFlavour.to_numpy(), abs(jet_flat[jet_light].eta).to_numpy(), jet_flat[jet_light].pt_nom.to_numpy())
        #array_heavy = self._btag_sf["deepJet_comb"].evaluate("central", "L", jet_flat[jet_heavy].hadronFlavour.to_numpy(), abs(jet_flat[jet_heavy].eta).to_numpy(), jet_flat[jet_heavy].pt_nom.to_numpy())
        #btagSF_deepjet_L[jet_light] = array_light
        #btagSF_deepjet_L[jet_heavy] = array_heavy
        #btagSF_deepjet_L = ak.unflatten(btagSF_deepjet_L, ak.num(emevents.Jet))
        #bTagSF = ak.prod(1-btagSF_deepjet_L, axis=1)

        #SF = bTagSF*emevents.puWeight*emevents.genWeight

        #PU/PF/Gen Weights
        #if self._year != '2018':
        #  SF = SF*emevents.L1PreFiringWeight.Nom

        #Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
        #Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
        #
        ##Muon SF
        #Muon_low = ak.mask(Muon_collections, Muon_collections['pt'] <= 120)
        #Muon_Hi = ak.mask(Muon_collections, Muon_collections['pt'] > 120)
        #Trk_SF = ak.fill_none(self._evaluator['trackerMu'](abs(Muon_low.eta), Muon_low.pt), 1)
        #Trk_SF_Hi = ak.fill_none(self._evaluator['trackerMu_Hi'](abs(Muon_Hi.eta), Muon_Hi.rho), 1)
        #if '2016' in self._year:
        #  triggerstr = 'NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight'
        #elif self._year == '2017':
        #  triggerstr = 'NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight'
        #elif self._year == '2018':
        #  triggerstr = 'NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight'
        #MuTrigger_SF = self._m_sf[triggerstr].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 
        #MuID_SF = self._m_sf["NUM_TightID_DEN_TrackerMuons"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 
        #MuISO_SF = self._m_sf["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 

        #SF = SF*MuTrigger_SF*MuID_SF*MuISO_SF*Trk_SF*Trk_SF_Hi

        ##Electron SF and lumi
        #EleReco_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","RecoAbove20", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
        #EleIDnoISO_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","wp80noiso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
        #SF = SF*EleReco_SF*EleIDnoISO_SF*self._lumiWeight[emevents.metadata["dataset"]]
        SF = emevents.genWeight#*self._lumiWeight[emevents.metadata["dataset"]]
        SF = SF.to_numpy()
        SF[abs(SF)>10] = 0
        emevents["weight"] = SF
        return emevents 

    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        #make interesting variables
        emVar = Electron_collections + Muon_collections
        emevents["e_m_Mass"] = emVar.mass
        return emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = self.Corrections(emevents)
          emevents = self.SF(emevents)
          emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
          out[f'hist'].fill(
              sample=emevents.metadata["dataset"].split('_')[-1],
              emmass=emevents[f'e_m_Mass'], 
              weight=emevents["weight"]
          )
        else:
          print("No Events found in "+emevents.metadata["dataset"]) 
        return out

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
  years = ['2016preVFP', '2016postVFP', '2017', '2018']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
    if '2016' in year:
      TrackerMu_Hi=["trackerMu_Hi NUM_TrackerMuons_DEN_genTracks/abseta_p_value Corrections/TrackerMu/2016HighPt.json"]
      if 'pre' in year:
        TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json"]
      else:
        TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json"]
         
    elif '2017' in year:
      TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json"]
      TrackerMu_Hi=["trackerMu_Hi NUM_TrackerMuons_DEN_genTracks/abseta_p_value Corrections/TrackerMu/2017HighPt.json"]
    elif '2018' in year:
      TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json"]
      TrackerMu_Hi=["trackerMu_Hi NUM_TrackerMuons_DEN_genTracks/abseta_p_value Corrections/TrackerMu/2018HighPt.json"]

    btag_sf = correctionlib.CorrectionSet.from_file(f"jsonpog-integration/POG/BTV/{year}_UL/btagging.json.gz")
    muon_sf = correctionlib.CorrectionSet.from_file(f"jsonpog-integration/POG/MUO/{year}_UL/muon_Z.json.gz")
    electron_sf = correctionlib.CorrectionSet.from_file(f"jsonpog-integration/POG/EGM/{year}_UL/electron.json.gz")
    ext = extractor()
    ext.add_weight_sets(TrackerMu)
    ext.add_weight_sets(TrackerMu_Hi)
    ext.finalize()
    evaluator = ext.make_evaluator()
    processor_instance = MyEMuPeak(lumiWeight, year, btag_sf, muon_sf, electron_sf, evaluator)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
