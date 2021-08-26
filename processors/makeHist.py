from coffea import processor, hist
from coffea.util import save
import awkward as ak
import numpy, json, os

class MyEMuPeak(processor.ProcessorABC):
    def __init__(self, lumiWeight, year):
        dataset_axis = hist.Cat("dataset", "samples")
        self._lumiWeight = lumiWeight
        self._year = year
        self._accumulator = processor.dict_accumulator({
            'MET': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("MET", r"$E^{miss}_{T}$ [GeV]", 25, 0, 250),
            ),
          #  'emMass': hist.Hist(
          #      "Events",
          #      dataset_axis,
          #      hist.Bin("emMass", r"$m^{e\mu}$ [GeV]", 50, 110, 160),
          #  ),
          #  'ePt': hist.Hist(
          #      "Events",
          #      dataset_axis,
          #      hist.Bin("ePt", r"$p^{e}_{T}$ [GeV]", 40, 24, 200),
          #  ),
          #  'mPt': hist.Hist(
          #      "Events",
          #      dataset_axis,
          #      hist.Bin("mPt", r"$p^{\mu}_{T}$ [GeV]", 40, 24, 200),
          #  ),
          #  'eEta': hist.Hist(
          #      "Events",
          #      dataset_axis,
          #      hist.Bin("eEta", r"$\eta^{e}$", 50, -2.5, 2.5),
          #  ),
          #  'mEta': hist.Hist(
          #      "Events",
          #      dataset_axis,
          #      hist.Bin("mEta", r"$\eta^{\mu}$", 50, -2.4, 2.4),
          #  ),
            'j1Pt': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("j1Pt", r"$p^{j_{1}}_{T}$ [GeV]", 40, 30, 400),
            ),
            'j2Pt': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("j2Pt", r"$p^{j_{2}}_{T}$ [GeV]", 40, 30, 200),
            ),
            'j1Eta': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("j1Eta", r"$\eta^{j_{1}}$", 30, -5, 5),
            ),
            'j2Eta': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("j2Eta", r"$\eta^{j_{2}}$", 30, -5, 5),
            ),
            'jEta-Pt': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("jEta", r"$\eta^{j}$", 30, -5, 5),
                hist.Bin("jPt", r"$p^{j}_{T}$ [GeV]", [30, 50, 100, 500], 30, 500)
            ),
            'j1Eta-Pt': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("j1Eta", r"$\eta^{j_{1}}$", 30, -5, 5),
                hist.Bin("j1Pt", r"$p^{j_{1}}_{T}$ [GeV]", [30, 50, 100, 500], 30, 500)
            ),
            'j2Eta-Pt': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("j2Eta", r"$\eta^{j_{2}}$", 30, -5, 5),
                hist.Bin("j2Pt", r"$p^{j_{2}}_{T}$ [GeV]", [30, 50, 100, 200], 30, 200)
            ),
            'jEta-Phi': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("jEta", r"$\eta^{j}$", 30, -5, 5),
                hist.Bin("jPhi", r"$\phi^{j}$", 20, -3.2, 3.2)
            ),
            'j1Eta-Phi': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("j1Eta", r"$\eta^{j_{1}}$", 30, -5, 5),
                hist.Bin("j1Phi", r"$\phi^{j_{1}}$", 20, -3.2, 3.2)
            ),
            'j2Eta-Phi': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("j2Eta", r"$\eta^{j_{2}}$", 30, -5, 5),
                hist.Bin("j2Phi", r"$\phi^{j_{2}}$", 20, -3.2, 3.2)
            ),
            'njets': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("njets", "Number of Jets", 10, 0, 10),
            ),
        #    'emumass2D': hist.Hist(
        #        "Events",
        #        dataset_axis,
        #        hist.Bin("emumass", "e_mu mass", 50, 110, 160),
        #        hist.Bin("ept", "e_pt", 40, 20, 200),
        #    ),
           
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
        trg_collections = trg_collections[(((trg_collections.filterBits >> 1) & 1)==1) & (trg_collections.id == 13)]

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
        elif emevents.metadata["dataset"] == 'SingleMuon' or emevents.metadata["dataset"] == 'data':
            massRange = ((emVar.mass<115) & (emVar.mass>110)) | ((emVar.mass<160) & (emVar.mass>135))
        else:
            massRange = (emVar.mass<160) & (emVar.mass>110)
            #massRange = ((emVar.mass<115) & (emVar.mass>110)) | ((emVar.mass<160) & (emVar.mass>135))
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	
 
    def SF(self, emevents):
        if emevents.metadata["dataset"]=='SingleMuon' or emevents.metadata["dataset"] == 'data': return numpy.ones(len(emevents))
        #Get bTag SF
        bTagSF_L = ak.prod(1-emevents.Jet.btagSF_deepjet_L*emevents.Jet.passDeepJet_L, axis=1)
        bTagSF_M = ak.prod(1-emevents.Jet.btagSF_deepjet_M*emevents.Jet.passDeepJet_M, axis=1)

        #PU/PF/Gen Weights
        SF = emevents.puWeight*emevents.PrefireWeight*emevents.genWeight

        Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
        
        #Muon SF
        SF = SF*Muon_collections.Trigger_SF*Muon_collections.ID_SF*Muon_collections.ISO_SF

        #Electron SF
        SF = SF*Electron_collections.Reco_SF*Electron_collections.ID_SF
        
        return SF*self._lumiWeight[emevents.metadata["dataset"]]
        
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = self.Corrections(emevents)
          weight = self.SF(emevents)
          emu = Muon_collections + Electron_collections
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
            sample_group_name = emevents.metadata["dataset"] #r'$H\rightarrow e\mu$ (BR=1%)'
          out['MET'].fill(
               dataset=sample_group_name,
               MET=MET_collections.pt, 
               weight=weight
          )
#          out['emMass'].fill(
#              dataset=sample_group_name,
#              emMass=emu.mass, 
#              weight=weight
#          )
#          out['ePt'].fill(
#              dataset=sample_group_name,
#              ePt=Electron_collections.pt, 
#              weight=weight
#          )
#          out['mPt'].fill(
#              dataset=sample_group_name,
#              mPt=Muon_collections.pt, 
#              weight=weight
#          )
#          out['eEta'].fill(
#              dataset=sample_group_name,
#              eEta=Electron_collections.eta, 
#              weight=weight
#          )
#          out['mEta'].fill(
#              dataset=sample_group_name,
#              mEta=Muon_collections.eta, 
#              weight=weight
#          )
          out['j1Pt'].fill(
              dataset=sample_group_name,
              j1Pt=Jet_collections[emevents.nJet30 >= 1][:,0].pt, 
              weight=weight[emevents.nJet30 >= 1]
          )
          out['j2Pt'].fill(
              dataset=sample_group_name,
              j2Pt=Jet_collections[emevents.nJet30 >= 2][:,1].pt, 
              weight=weight[emevents.nJet30 >= 2]
          )
          out['j1Eta'].fill(
              dataset=sample_group_name,
              j1Eta=Jet_collections[emevents.nJet30 >= 1][:,0].eta, 
              weight=weight[emevents.nJet30 >= 1]
          )
          out['j2Eta'].fill(
              dataset=sample_group_name,
              j2Eta=Jet_collections[emevents.nJet30 >= 2][:,1].eta, 
              weight=weight[emevents.nJet30 >= 2]
          )
          out['jEta-Pt'].fill(
              dataset=sample_group_name,
              jEta=ak.flatten(Jet_collections[emevents.nJet30 >= 1].eta), 
              jPt=ak.flatten(Jet_collections[emevents.nJet30 >= 1].pt), 
              weight=numpy.repeat(ak.to_numpy(weight[emevents.nJet30 >= 1]), ak.num(Jet_collections[emevents.nJet30 >= 1].pt)) 
          )
          out['j1Eta-Pt'].fill(
              dataset=sample_group_name,
              j1Eta=Jet_collections[emevents.nJet30 >= 1][:,0].eta, 
              j1Pt=Jet_collections[emevents.nJet30 >= 1][:,0].pt, 
              weight=weight[emevents.nJet30 >= 1]
          )
          out['j2Eta-Pt'].fill(
              dataset=sample_group_name,
              j2Eta=Jet_collections[emevents.nJet30 >= 2][:,1].eta, 
              j2Pt=Jet_collections[emevents.nJet30 >= 2][:,1].pt, 
              weight=weight[emevents.nJet30 >= 2]
          )
          out['jEta-Phi'].fill(
              dataset=sample_group_name,
              jEta=ak.flatten(Jet_collections[emevents.nJet30 >= 1].eta), 
              jPhi=ak.flatten(Jet_collections[emevents.nJet30 >= 1].phi), 
              weight=numpy.repeat(ak.to_numpy(weight[emevents.nJet30 >= 1]), ak.num(Jet_collections[emevents.nJet30 >= 1].eta)) 
          )
          out['j1Eta-Phi'].fill(
              dataset=sample_group_name,
              j1Eta=Jet_collections[emevents.nJet30 >= 1][:,0].eta, 
              j1Phi=Jet_collections[emevents.nJet30 >= 1][:,0].phi, 
              weight=weight[emevents.nJet30 >= 1]
          )
          out['j2Eta-Phi'].fill(
              dataset=sample_group_name,
              j2Eta=Jet_collections[emevents.nJet30 >= 2][:,1].eta, 
              j2Phi=Jet_collections[emevents.nJet30 >= 2][:,1].phi, 
              weight=weight[emevents.nJet30 >= 2]
          )
          out['njets'].fill(
              dataset=sample_group_name,
              njets=emevents.nJet30, 
              weight=weight
          )
          #out['emumass2D'].fill(
          #    dataset=emevents.metadata["dataset"],
          #    emumass=emu.mass, 
          #    ept=Electron_collections.pt, 
          #    weight=weight
          #)
        else:
          print("No Events found in "+emevents.metadata["dataset"]) 
        return out

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
  years = ['2017']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
    processor_instance = MyEMuPeak(lumiWeight, year)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
