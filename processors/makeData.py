from coffea import processor, hist
from coffea.util import save
import xgboost as xgb
import awkward as ak
import numpy, json, os
from kinematics import *
from Vetos import *
from Corrections import *
from BDT_functions import *

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
    
    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = Vetos(self._year, events)
        if len(emevents)==0: return out
        emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = Corrections(emevents)
        if len(emevents)==0: return out
        emevents['weight'] = ak.sum(emevents.Jet.passDeepJet_L,1)==0 
        emevents = interestingKin(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
        BDT_fun = BDT_functions(self._BDTmodels, self.var_GG_, self.var_2jet_VBF_)
        emevents = BDT_fun.pandasDF(emevents)
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
