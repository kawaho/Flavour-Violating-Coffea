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
            'mass_pt': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("mmmass", r"$m^{\mu\mu}$ [GeV]", [50,100,200,500,1000], 50, 1000),
                hist.Bin("mmpt", r"$p^{\mu\mu}_{T}$ [GeV]", [0,10,20,30,40,50,100,150,200,300,400,1000], 0, 1000),
            ),
           
        })

    @property
    def accumulator(self):
        return self._accumulator
    
    def Vetos(self, events):
        #Choose mm channel and single muon triggers

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

        mmevents = events[(events.channel == 1) & (trigger == 1)]

        M_collections = mmevents.Muon

        #Kinematics Selections
        mmevents["Muon", "Target"] = ((M_collections.pt > 15) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.045) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

        mmevents = mmevents[ak.sum(mmevents.Muon.Target, axis=1)==2]
        Muon_collections = mmevents.Muon[mmevents.Muon.Target==1]

        #Opposite Charge
        opp_charge = (Muon_collections[:,0].charge*Muon_collections[:,1].charge) == -1

        mmevents = mmevents[opp_charge]
        Muon_collections = mmevents.Muon[mmevents.Muon.Target==1]

        #Trig Matching
        trg_collections = mmevents.TrigObj

        trg_collections = trg_collections[(((trg_collections.filterBits >> 1) & 1)==1) & (trg_collections.id == 13)]

        trg_Match1 = ak.any((Muon_collections[:,0].delta_r(trg_collections) < 0.5),1) & (Muon_collections[:,0].pt>mpt_threshold)
        trg_Match2 = ak.any((Muon_collections[:,1].delta_r(trg_collections) < 0.5),1) & (Muon_collections[:,1].pt>mpt_threshold)

        mmevents = mmevents[trg_Match1 | trg_Match2]
        mmevents.Muon['pt'] = mmevents.Muon['corrected_pt']

        return mmevents
   
    def SF(self, mmevents, Muon_collections):
        if mmevents.metadata["dataset"]=='data': return numpy.ones(len(mmevents))

        #PU/PF/Gen Weights
        SF = mmevents.puWeight*mmevents.PrefireWeight*mmevents.genWeight

        M1_collections = Muon_collections[:,0]
        M2_collections = Muon_collections[:,1]
        
        #Muon 1 SF
        SF = SF*M1_collections.Trigger_SF*M1_collections.ID_SF*M1_collections.ISO_SF

        #Muon 2 SF
        SF = SF*M2_collections.Trigger_SF*M2_collections.ID_SF*M2_collections.ISO_SF
        
        return SF*self._lumiWeight[mmevents.metadata["dataset"]]
        
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        mmevents = self.Vetos(events)
        if len(mmevents)>0:
          Muon_collections = mmevents.Muon[mmevents.Muon.Target==1]
          mm_collections = Muon_collections[:,0]+Muon_collections[:,1]
          weight = self.SF(mmevents, Muon_collections)
          out['mass_pt'].fill(
              dataset=mmevents.metadata["dataset"],
              mmmass=mm_collections.mass, 
              mmpt=mm_collections.pt, 
              weight=weight
          )
        else:
          print("No Events found in "+mmevents.metadata["dataset"]) 
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
