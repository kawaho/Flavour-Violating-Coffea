from coffea import processor, hist
from coffea.util import save
from coffea.lookup_tools import extractor
import awkward as ak
import numpy, json, os

class MyEMuPeak(processor.ProcessorABC):
    def __init__(self, lumiWeight, year):
        dataset_axis = hist.Cat("dataset", "samples")
        self._lumiWeight = lumiWeight
        self._year = year
        ext = extractor()
        ext.add_weight_sets([f'Zptm Zptm ../DYreweight/Zptm_{year}.root'])
        ext.finalize()
        self.evaluator = ext.make_evaluator()['Zptm']

        self._accumulator = processor.dict_accumulator({
            'mmmass': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("mmmass", r"$m^{\mu\mu}$ [GeV]", [50,100,200,500,1000], 50, 1000),
            ),
            'mmpt': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("mmpt", r"$p^{\mu\mu}_{T}$ [GeV]", [0,10,20,30,40,50,100,150,200,300,400,1000], 0, 1000),
            ),
            'mmmass_corrected': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("mmmass_corrected", r"$m^{\mu\mu}$ [GeV]", [50,100,200,500,1000], 50, 1000),
            ),
            'mmpt_corrected': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("mmpt_corrected", r"$p^{\mu\mu}_{T}$ [GeV]", [0,10,20,30,40,50,100,150,200,300,400,1000], 0, 1000),
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
    
        #lumiWeight
        SF*self._lumiWeight[mmevents.metadata["dataset"]]

	if ('DY' in mmevents.metadata["dataset"] and not '10to50' in mmevents.metadata["dataset"]):
          SF_corrected = SF*self.evaluator(mmevents.genZ_M, mmevents.genZ_pt)
        else:
          SF_corrected = SF
     
        return SF, SF_corrected
        
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        mmevents = self.Vetos(events)
        if len(mmevents)>0:
          Muon_collections = mmevents.Muon[mmevents.Muon.Target==1]
          mm_collections = Muon_collections[:,0]+Muon_collections[:,1]
          weight, weight_corrected = self.SF(mmevents, Muon_collections)
          sample_group_name = "" 
          if "ST" in emevents.metadata["dataset"] or "TT" in emevents.metadata["dataset"]:
            sample_group_name = r'$t\bar{t}$,t+Jets'
          elif "HTo" in emevents.metadata["dataset"] and not "LFV" in emevents.metadata["dataset"]:
            sample_group_name = 'SM Higgs'
          elif "ZZ" in emevents.metadata["dataset"] or "WZ" in emevents.metadata["dataset"] or "WZ" in emevents.metadata["dataset"]:
            sample_group_name = "Diboson"
          elif "DY" in emevents.metadata["dataset"]:
            sample_group_name = "DY+Jets"
          elif "WJ" in emevents.metadata["dataset"] or "WG" in emevents.metadata["dataset"]:
            sample_group_name = "W+Jets"
          elif "EWK" in emevents.metadata["dataset"]:
            sample_group_name = "EWK W/Z"
          elif "data" in emevents.metadata["dataset"]:
            sample_group_name = "data"
          out['mmmass'].fill(
              dataset=sample_group_name,
              mmmass=mm_collections.mass, 
              weight=weight
          )
          out['mmpt'].fill(
              dataset=sample_group_name,
              mmpt=mm_collections.pt, 
              weight=weight
          )
          out['mmmass_corrected'].fill(
              dataset=sample_group_name,
              mmmass_corrected=mm_collections.mass, 
              weight=weight_corrected
          )
          out['mmpt_corrected'].fill(
              dataset=sample_group_name,
              mmpt_corrected=mm_collections.pt, 
              weight=weight_corrected
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
