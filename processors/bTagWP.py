from coffea import processor, hist
from coffea.util import save
import awkward as ak
import numpy, json, os

class MyEMuPeak(processor.ProcessorABC):
    def __init__(self, lumiWeight):
        dataset_axis = hist.Cat("dataset", "samples")
        self._lumiWeight = lumiWeight
        self._accumulator = processor.dict_accumulator({
            'emMass': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("emMass", r"$m^{e\mu}$ [GeV]", 50, 110, 160),
            ),
            'emMass_deepjet_L': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("emMass_deepjet_L", r"$m^{e\mu}$ [GeV]", 50, 110, 160),
            ),
            'emMass_deepjet_M': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("emMass_deepjet_M", r"$m^{e\mu}$ [GeV]", 50, 110, 160),
            )
#            'emMass_deepcsv_L': hist.Hist(
#                "Events",
#                dataset_axis,
#                hist.Bin("emMass_deepcsv_L", r"$m_{e\mu}$ [GeV]", 50, 110, 160),
#            ),
#            'emMass_deepcsv_M': hist.Hist(
#                "Events",
#                dataset_axis,
#                hist.Bin("emMass_deepcsv_M", r"$m_{e\mu}$ [GeV]", 50, 110, 160),
#            )
        })

    @property
    def accumulator(self):
        return self._accumulator
    
    def Vetos(self, events):
        #Choice em channel and Iso27
        emevents = events[(events.channel == 0) & (events.HLT.IsoMu27 == 1)]

        E_collections = emevents.Electron
        M_collections = emevents.Muon

        #Kinematics Selections
        emevents["Electron", "Target"] = ((E_collections.pt > 24) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.045) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
        emevents["Muon", "Target"] = ((M_collections.pt > 29) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.045) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

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

        #Muon pT corrections
        Muon_collections['pt'] = Muon_collections['corrected_pt']

        #Take the first leptons
        Electron_collections = Electron_collections[:,0]
        Muon_collections = Muon_collections[:,0]
        emVar = Electron_collections + Muon_collections

        if 'LFV' in emevents.metadata["dataset"]:
            massRange = (emVar.mass<135) & (emVar.mass>115)
        elif emevents.metadata["dataset"] == 'data':
            massRange = ((emVar.mass<115) & (emVar.mass>110)) | ((emVar.mass<160) & (emVar.mass>135))
        else:
            massRange = (emVar.mass<135) & (emVar.mass>115) #((emVar.mass<115) & (emVar.mass>110)) | ((emVar.mass<160) & (emVar.mass>135))
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange]	
 
    def SF(self, emevents):
        if emevents.metadata["dataset"]=='data': return numpy.ones(len(emevents)), ak.sum(emevents.Jet.passDeepJet_L,1)==0, ak.sum(emevents.Jet.passDeepJet_M,1)==0#, ak.sum(emevents.Jet.passDeepCSV_L,1)==0, ak.sum(emevents.Jet.passDeepCSV_M,1)==0
        #Get bTag SF
        bTagJetSF_L = ak.prod(1-emevents.Jet.btagSF_deepjet_L*emevents.Jet.passDeepJet_L, axis=1)
        bTagJetSF_M = ak.prod(1-emevents.Jet.btagSF_deepjet_M*emevents.Jet.passDeepJet_M, axis=1)
#        bTagCSVSF_L = ak.prod(1-emevents.Jet.btagSF_deepcsv_L*emevents.Jet.passDeepCSV_L, axis=1)
#        bTagCSVSF_M = ak.prod(1-emevents.Jet.btagSF_deepcsv_M*emevents.Jet.passDeepCSV_M, axis=1)

        #PU/PF/Gen Weights
        SF = emevents.puWeight*emevents.PrefireWeight*emevents.genWeight

        Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
        
        #Muon SF
        SF = SF*Muon_collections.Trigger_SF*Muon_collections.ID_SF*Muon_collections.ISO_SF

        #Electron SF
        SF = SF*Electron_collections.Reco_SF*Electron_collections.ID_SF*self._lumiWeight[emevents.metadata["dataset"]]
        
        return SF, SF*bTagJetSF_L, SF*bTagJetSF_M#, SF*bTagCSVSF_L, SF*bTagCSVSF_M
        
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections = self.Corrections(emevents)
          weight, weight_deepjet_L, weight_deepjet_M = self.SF(emevents)
          #weight_deepjet_L, weight_deepjet_M, weight_deepcsv_L, weight_deepcsv_M = self.SF(emevents)
          emu = Muon_collections + Electron_collections
          out['emMass'].fill(
              dataset=emevents.metadata["dataset"],
              emMass=emu.mass, 
              weight=weight
          )
          out['emMass_deepjet_L'].fill(
              dataset=emevents.metadata["dataset"],
              emMass_deepjet_L=emu.mass, 
              weight=weight_deepjet_L
          )
          out['emMass_deepjet_M'].fill(
              dataset=emevents.metadata["dataset"],
              emMass_deepjet_M=emu.mass, 
              weight=weight_deepjet_M
          )
#          out['emMass_deepcsv_L'].fill(
#              dataset=emevents.metadata["dataset"],
#              emMass_deepcsv_L=emu.mass, 
#              weight=weight_deepcsv_L
#          )
#          out['emMass_deepcsv_M'].fill(
#              dataset=emevents.metadata["dataset"],
#              emMass_deepcsv_M=emu.mass, 
#              weight=weight_deepcsv_M
#          )
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
    processor_instance = MyEMuPeak(lumiWeight)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
