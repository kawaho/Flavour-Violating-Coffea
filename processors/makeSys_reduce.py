from coffea import processor, hist
from coffea.util import save
from coffea.lookup_tools import extractor
from coffea.btag_tools import BTagScaleFactor
import xgboost as xgb
import pandas as pd
import awkward as ak
import correctionlib, numpy, json, os
from kinematics import *
from Vetos import *
from Corrections import *
from BDT_functions import *

class MyEMuPeak(processor.ProcessorABC):
    def __init__(self, lumiWeight, BDTmodels, BDTvars, year, btag_sf, muon_sf, electron_sf, evaluator, VBF_quan, GG_quan):
        self._lumiWeight = lumiWeight
        self._BDTmodels = BDTmodels
        self._btag_sf = btag_sf
        self._e_sf = electron_sf
        self._m_sf = muon_sf
        self._evaluator = evaluator
        self._year = year
        self._jecYear = self._year[:4]
        self.var_GG_ = BDTvars['model_GG']
        self.var_2jet_VBF_ = BDTvars['model_VBF']
        self.jetUnc = ['jesAbsolute', 'jesBBEC1', 'jesFlavorQCD', 'jesEC2', 'jesHF', 'jesRelativeBal']
        self.metUnc = ['UnclusteredEn']
        self.jetyearUnc = sum([[f'jer_{year}', f'jesAbsolute_{year}', f'jesBBEC1_{year}', f'jesEC2_{year}', f'jesHF_{year}', f'jesRelativeSample_{year}'] for year in ['2017', '2018', '2016']], [])
        self.sfUnc = sum([[f'pu_{year}', f'bTag_{year}'] for year in ['2017', '2018', '2016']], [])
        self.sfUnc += ['pf_2016', 'pf_2017', 'mID', 'mIso', 'mTrg', 'eReco', 'eID']
        self.theoUnc = [f'lhe{i}' for i in range(103)] + ['scalep5p5', 'scale22']
        self.leptonUnc = ['ees', 'eer', 'me']
        self._accumulator = processor.dict_accumulator({})
        self._accumulator[f'e_m_Mass'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'isVBFcat'] = processor.column_accumulator(numpy.bool_([]))
        self._accumulator[f'isVBF'] = processor.column_accumulator(numpy.bool_([]))
        self._accumulator[f'isHerwig'] = processor.column_accumulator(numpy.bool_([]))
        self._accumulator[f'njets'] = processor.column_accumulator(numpy.ubyte([]))
        self._accumulator[f'mva'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'year'] = processor.column_accumulator(numpy.ubyte([]))
        self._accumulator[f'weight'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'mva_hist-herwig_GGcat'] = hist.Hist("Events", hist.Bin("mva", "mva", GG_quan))
        self._accumulator[f'mva_hist-herwig_VBFcat'] = hist.Hist("Events", hist.Bin("mva", "mva", VBF_quan))
        for cat in ['GG_GGcat', 'GG_VBFcat', 'VBF_GGcat', 'VBF_VBFcat']:
          if 'VBFcat' in cat:
            self._accumulator[f'mva_hist-{cat}'] = hist.Hist("Events", hist.Bin("mva", "mva", VBF_quan))
          else:
            self._accumulator[f'mva_hist-{cat}'] = hist.Hist("Events", hist.Bin("mva", "mva", GG_quan))
          for sys in self.jetUnc+self.jetyearUnc+self.metUnc+self.sfUnc+self.leptonUnc:
              if 'VBFcat' in cat:
                self._accumulator[f'mva_hist_{sys}_Up-{cat}'] = hist.Hist("Events", hist.Bin("mva", "mva", VBF_quan))
                self._accumulator[f'mva_hist_{sys}_Down-{cat}'] = hist.Hist("Events", hist.Bin("mva", "mva", VBF_quan))
              else:
                self._accumulator[f'mva_hist_{sys}_Up-{cat}'] = hist.Hist("Events", hist.Bin("mva", "mva", GG_quan))
                self._accumulator[f'mva_hist_{sys}_Down-{cat}'] = hist.Hist("Events", hist.Bin("mva", "mva", GG_quan))
          for sys in self.theoUnc:
              if 'VBFcat' in cat:
                self._accumulator[f'mva_hist_{sys}-{cat}'] = hist.Hist("Events", hist.Bin("mva", "mva", VBF_quan))
              else:
                self._accumulator[f'mva_hist_{sys}-{cat}'] = hist.Hist("Events", hist.Bin("mva", "mva", GG_quan))
        for sys in self.leptonUnc:
            self._accumulator[f'mva_{sys}_Up'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[f'mva_{sys}_Down'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[f'e_m_Mass_{sys}_Up'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[f'e_m_Mass_{sys}_Down'] = processor.column_accumulator(numpy.array([]))

    @property
    def accumulator(self):
        return self._accumulator
    
    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        #make interesting variables
        #zero/any no. of jets
        if '2016' in self._year:
          emevents["year"] = numpy.full(len(emevents), 0, dtype=numpy.ubyte)
        elif '2017' in self._year:
          emevents["year"] = numpy.full(len(emevents), 1, dtype=numpy.ubyte)
        else:
          emevents["year"] = numpy.full(len(emevents), 2, dtype=numpy.ubyte)
        if ('VBF' in emevents.metadata["dataset"]) and (not 'H' in emevents.metadata["dataset"][-1]):
          emevents["isVBF"] = numpy.full(len(emevents), True, dtype=numpy.bool_) 
        else:
          emevents["isVBF"] = numpy.full(len(emevents), False, dtype=numpy.bool_) 
        if 'H' in emevents.metadata["dataset"][-1]:
          emevents["isHerwig"] = numpy.full(len(emevents), True, dtype=numpy.bool_) 
        else:
          emevents["isHerwig"] = numpy.full(len(emevents), False, dtype=numpy.bool_)
        return emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = Vetos(self._year, events)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = Corrections(emevents)
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = Corrections(emevents)
          SF_fun = SF(self._lumiWeight, self._year, self._btag_sf, self._m_sf, self._e_sf, self._evaluator)
          emevents = SF_fun.evaluate(emevents, doQCD=False, doSys=True)
          emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
          emevents = interestingKin(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections, doSys=True)
          BDT_fun = BDT_functions(self._BDTmodels, self.var_GG_, self.var_2jet_VBF_)
          emevents = BDT_fun.pandasDF(emevents)
          for sys in self.leptonUnc+self.metUnc:
            emevents = BDT_fun.pandasDF(emevents, sys, 'Up')
            emevents = BDT_fun.pandasDF(emevents, sys, 'Down')

          for sys in self.jetUnc+self.jetyearUnc:
            emevents = BDT_fun.pandasDF(emevents, sys, 'Up', True)
            emevents = BDT_fun.pandasDF(emevents, sys, 'Down', True)

          for sys_var_ in out:
            if '_hist' in sys_var_:
              continue
            else:
              acc = emevents[sys_var_].to_numpy()
              out[sys_var_].add( processor.column_accumulator( acc ) )

          if 'H' in emevents.metadata["dataset"][-1]: 
            out[f'mva_hist-herwig_GGcat'].fill(
                mva=emevents[emevents.isVBFcat==0][f'mva'], 
                weight=emevents[emevents.isVBFcat==0]["weight"]
            )
            out[f'mva_hist-herwig_VBFcat'].fill(
                mva=emevents[emevents.isVBFcat==1][f'mva'], 
                weight=emevents[emevents.isVBFcat==1]["weight"]
            )
          else:
            out[f'mva_hist-GG_GGcat'].fill(
                mva=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==0)][f'mva'], 
                weight=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==0)]["weight"]
            )
            out[f'mva_hist-GG_VBFcat'].fill(
                mva=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==1)][f'mva'], 
                weight=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==1)]["weight"]
            )
            out[f'mva_hist-VBF_GGcat'].fill(
                mva=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==0)][f'mva'], 
                weight=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==0)]["weight"]
            )
            out[f'mva_hist-VBF_VBFcat'].fill(
                mva=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==1)][f'mva'], 
                weight=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==1)]["weight"]
            )
            
            for sys in self.jetUnc+self.jetyearUnc:
              for UpDown in ['Up', 'Down']:
                out[f'mva_hist_{sys}_{UpDown}-GG_GGcat'].fill(
                    mva=emevents[(emevents.isVBF==0) & (emevents[f'isVBFcat_{sys}_{UpDown}']==0)][f'mva_{sys}_{UpDown}'], 
                    weight=emevents[(emevents.isVBF==0) & (emevents[f'isVBFcat_{sys}_{UpDown}']==0)]["weight"]
                )
                out[f'mva_hist_{sys}_{UpDown}-GG_VBFcat'].fill(
                    mva=emevents[(emevents.isVBF==0) & (emevents[f'isVBFcat_{sys}_{UpDown}']==1)][f'mva_{sys}_{UpDown}'], 
                    weight=emevents[(emevents.isVBF==0) & (emevents[f'isVBFcat_{sys}_{UpDown}']==1)]["weight"]
                )
                out[f'mva_hist_{sys}_{UpDown}-VBF_GGcat'].fill(
                    mva=emevents[(emevents.isVBF==1) & (emevents[f'isVBFcat_{sys}_{UpDown}']==0)][f'mva_{sys}_{UpDown}'], 
                    weight=emevents[(emevents.isVBF==1) & (emevents[f'isVBFcat_{sys}_{UpDown}']==0)]["weight"]
                )
                out[f'mva_hist_{sys}_{UpDown}-VBF_VBFcat'].fill(
                    mva=emevents[(emevents.isVBF==1) & (emevents[f'isVBFcat_{sys}_{UpDown}']==1)][f'mva_{sys}_{UpDown}'], 
                    weight=emevents[(emevents.isVBF==1) & (emevents[f'isVBFcat_{sys}_{UpDown}']==1)]["weight"]
                )
  
            for sys in self.metUnc+self.leptonUnc:
              for UpDown in ['Up', 'Down']:
                out[f'mva_hist_{sys}_{UpDown}-GG_GGcat'].fill(
                    mva=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==0)][f'mva_{sys}_{UpDown}'], 
                    weight=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==0)]["weight"]
                )
                out[f'mva_hist_{sys}_{UpDown}-GG_VBFcat'].fill(
                    mva=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==1)][f'mva_{sys}_{UpDown}'], 
                    weight=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==1)]["weight"]
                )
                out[f'mva_hist_{sys}_{UpDown}-VBF_GGcat'].fill(
                    mva=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==0)][f'mva_{sys}_{UpDown}'], 
                    weight=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==0)]["weight"]
                )
                out[f'mva_hist_{sys}_{UpDown}-VBF_VBFcat'].fill(
                    mva=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==1)][f'mva_{sys}_{UpDown}'], 
                    weight=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==1)]["weight"]
                )
            for sys in self.sfUnc:
              for UpDown in ['Up', 'Down']:
                out[f'mva_hist_{sys}_{UpDown}-GG_GGcat'].fill(
                    mva=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==0)][f'mva'], 
                    weight=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==0)][f"weight_{sys}_{UpDown}"]
                )
                out[f'mva_hist_{sys}_{UpDown}-GG_VBFcat'].fill(
                    mva=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==1)][f'mva'], 
                    weight=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==1)][f"weight_{sys}_{UpDown}"]
                )
                out[f'mva_hist_{sys}_{UpDown}-VBF_GGcat'].fill(
                    mva=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==0)][f'mva'], 
                    weight=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==0)][f"weight_{sys}_{UpDown}"]
                )
                out[f'mva_hist_{sys}_{UpDown}-VBF_VBFcat'].fill(
                    mva=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==1)][f'mva'], 
                    weight=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==1)][f"weight_{sys}_{UpDown}"]
                )
            for sys in self.theoUnc:
              out[f'mva_hist_{sys}-GG_GGcat'].fill(
                  mva=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==0)][f'mva'], 
                  weight=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==0)][f"weight_{sys}"]
              )
              out[f'mva_hist_{sys}-GG_VBFcat'].fill(
                  mva=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==1)][f'mva'], 
                  weight=emevents[(emevents.isVBF==0) & (emevents.isVBFcat==1)][f"weight_{sys}"]
              )
              out[f'mva_hist_{sys}-VBF_GGcat'].fill(
                  mva=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==0)][f'mva'], 
                  weight=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==0)][f"weight_{sys}"]
              )
              out[f'mva_hist_{sys}-VBF_VBFcat'].fill(
                  mva=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==1)][f'mva'], 
                  weight=emevents[(emevents.isVBF==1) & (emevents.isVBFcat==1)][f"weight_{sys}"]
              )

        else:
          print("No Events found in "+emevents.metadata["dataset"]) 
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
    VBF_quan = numpy.load(f"results/SenScan/VBFcat_quantiles",allow_pickle=True)
    GG_quan = numpy.load(f"results/SenScan/GGcat_quantiles",allow_pickle=True)
    processor_instance = MyEMuPeak(lumiWeight, BDTmodels, BDTvars, year, btag_sf, muon_sf, electron_sf, evaluator, VBF_quan, GG_quan)
    outname = os.path.basename(__file__).replace('.py','')
    for masspt in [120,125,130]:
      save(processor_instance, f'processors/{outname}_{masspt}_{year}.coffea')
