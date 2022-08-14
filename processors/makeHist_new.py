from coffea import processor, hist
from coffea.util import save
#from coffea.btag_tools import BTagScaleFactor
import awkward as ak
import numpy, json, os

class MyEMuPeak(processor.ProcessorABC):
    def __init__(self, lumiWeight, year):
        dataset_axis = hist.Cat("dataset", "samples")
        self._lumiWeight = lumiWeight
        #self._btag_sf = btag_sf
        self._year = year
        self._accumulator = processor.dict_accumulator({
            'MET': hist.Hist(
                "Events",
                dataset_axis,
                hist.Bin("MET", r"$E^{miss}_{T}$ [GeV]", 25, 0, 250),
            ),
        } for )

    @property
    def accumulator(self):
        return self._accumulator
    
    def process(self, events):
        out = self.accumulator.identity()
        sameCharge = not ('LFV' in events.metadata["dataset"])
        emevents = Vetos(self._year, events, sameCharge=sameCharge)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = Corrections(emevents, (90,180))
          SF_fun = SF(self._lumiWeight, self._year, self._btag_sf, self._m_sf, self._e_sf, self._e_sf_pri, self._evaluator)
          emevents = SF_fun.evaluate(emevents, doQCD=sameCharge)
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = emevents[emevents.weight!=0], Electron_collections[emevents.weight!=0], Muon_collections[emevents.weight!=0], MET_collections[emevents.weight!=0], Jet_collections[emevents.weight!=0]

          emevents = interestingKin(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
          emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)

          for var in self.var_+self.var_1jet_+self.var_2jet_ :
            new_array = emevents[var].to_numpy()
            if type(new_array)==numpy.ma.core.MaskedArray:
              new_array = new_array.filled(numpy.nan)
            out[var].add( processor.column_accumulator( new_array ) )


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
            sample_group_name = r'$H\rightarrow e\mu$ (BR=1%)'
          out['MET'].fill(
               dataset=sample_group_name,
               MET=MET_collections.pt, 
               weight=weight
          )
        else:
          print("No Events found in "+emevents.metadata["dataset"]) 
        return out

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
  years = ['2017','2018']
#  bTagFiles = ['bTagFiles/DeepJet_106XUL17SF_WPonly_V2p1.csv','bTagFiles/DeepJet_106XUL18SF_WPonly.csv']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
#    btag_sf = BTagScaleFactor(bTagFile, "MEDIUM")
    processor_instance = MyEMuPeak(lumiWeight, year)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
