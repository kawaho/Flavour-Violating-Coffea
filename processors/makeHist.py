from coffea import processor, hist
from coffea.util import save
import awkward as ak
import numpy, json, os

class MyEMuPeak(processor.ProcessorABC):
    def __init__(self, lumiWeight):
        dataset_axis = hist.Cat("dataset", "samples")
        self._lumiWeight = lumiWeight
        self._accumulator = processor.dict_accumulator({
            'emumass': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("emumass", "e_mu mass", 50, 110, 160),
            ),
            'emumass2D': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("emumass", "e_mu mass", 50, 110, 160),
                hist.Bin("ept", "e_pt", 40, 20, 200),
            ),
           
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
        emevents["Electron", "Target"] = ((E_collections.pt > 24) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.045) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1))
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
        trg_collections = trg_collections[(((trg_collections.filterBits >> 1) & 1)==1) & (trg_collections.id == 13) & (trg_collections.pt > 29) & (ak.num(M_collections) == 1)]

        trg_Match = ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1)

        return emevents[trg_Match]
    
    def SF(self, emevents):
        if emevents.metadata["dataset"]=='data': return numpy.ones(len(emevents))
        #Get bTag SF
        bTagSF_L = ak.prod(1-emevents.Jet.btagSF_deepjet_L*emevents.Jet.passDeepJet_L, axis=1)
        bTagSF_M = ak.prod(1-emevents.Jet.btagSF_deepjet_M*emevents.Jet.passDeepJet_M, axis=1)

        #PU/PF/Gen Weights
        SF = emevents.puWeight*emevents.PrefireWeight*emevents.genWeight

        Muon_collections = emevents.Muon[emevents.Muon.Target==1]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1]
        
        #Muon SF
        SF = SF*Muon_collections.Trigger_SF*Muon_collections.ID_SF*Muon_collections.ISO_SF

        #Electron SF
        SF = SF*Electron_collections.Reco_SF*Electron_collections.ID_SF
        
        return ak.flatten(SF)*self._lumiWeight[emevents.metadata["dataset"]]
        
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        emu = emevents.Muon[:,0] + emevents.Electron[:,0]
        weight = self.SF(emevents)
        out['emumass'].fill(
            dataset=emevents.metadata["dataset"],
            emumass=emu.mass, 
            weight=weight
        )
        out['emumass2D'].fill(
            dataset=emevents.metadata["dataset"],
            emumass=emu.mass, 
            ept=emevents.Electron[:,0].pt, 
            weight=weight
        )
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
