from coffea import processor, hist
from coffea.util import save
from coffea.lookup_tools import extractor
import awkward as ak
import correctionlib, numpy, json, os, sys
from kinematics import *
from Corrections import *

class SF:
    def __init__(self, lumiWeight, year, btag_sf, muon_sf, electron_sf, evaluator):
        self._lumiWeight = lumiWeight
        self._year = year
        self._btag_sf = btag_sf
        self._e_sf = electron_sf
        self._m_sf = muon_sf
        self._evaluator = evaluator
        self._jecYear = self._year[:4]
    
    def evaluate(self, emevents, doQCD=False, doSys=False):
        Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
          
        if emevents.metadata["dataset"]=='SingleMuon' or emevents.metadata["dataset"] == 'data': 
          SF = ak.sum(emevents.Jet.passDeepJet_L,1)==0 
        else:
          jet_flat = ak.flatten(emevents.Jet)
          btagSF_deepjet_L = numpy.zeros(len(jet_flat))
          jet_light = ak.where((jet_flat.passDeepJet_L) & (jet_flat.hadronFlavour==0))
          jet_heavy = ak.where((jet_flat.passDeepJet_L) & (jet_flat.hadronFlavour!=0))
          array_light = self._btag_sf["deepJet_incl"].evaluate("central", "L", jet_flat[jet_light].hadronFlavour.to_numpy(), abs(jet_flat[jet_light].eta).to_numpy(), jet_flat[jet_light].pt_nom.to_numpy())
          array_heavy = self._btag_sf["deepJet_comb"].evaluate("central", "L", jet_flat[jet_heavy].hadronFlavour.to_numpy(), abs(jet_flat[jet_heavy].eta).to_numpy(), jet_flat[jet_heavy].pt_nom.to_numpy())
          btagSF_deepjet_L[jet_light] = array_light
          btagSF_deepjet_L[jet_heavy] = array_heavy
          btagSF_deepjet_L = ak.unflatten(btagSF_deepjet_L, ak.num(emevents.Jet))
          bTagSF = ak.prod(1-btagSF_deepjet_L, axis=1)
    
          #bTag/PU/Gen Weights
          SF = bTagSF*emevents.puWeight*emevents.genWeight
    
          #PU/PF/Gen Weights
          if self._year != '2018':
            SF = SF*emevents.L1PreFiringWeight.Nom

          #Zvtx
          #if self._year == '2017':
          #  SF = 0.991*SF
    
          #Muon SF
          Muon_low = ak.mask(Muon_collections, Muon_collections['pt'] <= 120)
          Muon_Hi = ak.mask(Muon_collections, Muon_collections['pt'] > 120)
          Trk_SF = ak.fill_none(self._evaluator['trackerMu'](abs(Muon_low.eta), Muon_low.pt), 1)
          Trk_SF_Hi = ak.fill_none(self._evaluator['trackerMu_Hi'](abs(Muon_Hi.eta), Muon_Hi.rho), 1)
    
          if '2016' in self._year:
            triggerstr = 'NUM_IsoMu24_or_IsoTkMu24_DEN_CutBasedIdTight_and_PFIsoTight'
          elif self._year == '2017':
            triggerstr = 'NUM_IsoMu27_DEN_CutBasedIdTight_and_PFIsoTight'
          elif self._year == '2018':
            triggerstr = 'NUM_IsoMu24_DEN_CutBasedIdTight_and_PFIsoTight'

          Muon_pass = ak.mask(Muon_collections, emevents['mtrigger'])
          
          MuTrigger_SF = ak.where(emevents['mtrigger'], self._m_sf[triggerstr].evaluate(f"{self._year}_UL", abs(ak.fill_none(Muon_pass.eta,2)).to_numpy(), ak.fill_none(Muon_pass.pt,30).to_numpy(), "sf"), numpy.ones(len(SF))) 
          MuID_SF = self._m_sf["NUM_TightID_DEN_TrackerMuons"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 
          MuISO_SF = self._m_sf["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 
    
          SF = SF*MuTrigger_SF*MuID_SF*MuISO_SF*Trk_SF*Trk_SF_Hi
    
          #Electron SF and lumi
          EleReco_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","RecoAbove20", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
          EleIDnoISO_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","wp80noiso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
          SF = SF*EleReco_SF*EleIDnoISO_SF*self._lumiWeight[emevents.metadata["dataset"]]
    
          SF = SF.to_numpy()
          SF[abs(SF)>10] = 0
        emevents["weight"] = SF
        return emevents

class MyDF(processor.ProcessorABC):
    def __init__(self, lumiWeight, year, btag_sf, muon_sf, electron_sf, evaluator):
        self._samples = []
        self._lumiWeight = lumiWeight
        self._year = year
        self._btag_sf = btag_sf
        self._e_sf = electron_sf
        self._m_sf = muon_sf
        self._evaluator = evaluator
        self._accumulator = processor.dict_accumulator({})
        self.var_ = ["sample", "label", "weight", "mpt", "ept", "mtrigger", "etrigger"]
        self.var_1jet_ = []
        self.var_2jet_ = []
        for var in self.var_ :
            self._accumulator[var+'_0jets'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[var+'_1jets'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))
        for var in self.var_1jet_ :
            self._accumulator[var+'_1jets'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))
        for var in self.var_2jet_ :
            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))

    def sample_list(self, *argv):
        self._samples = argv

    @property
    def accumulator(self):
        return self._accumulator

    def Vetos(self, _year, events, sameCharge=False):
        if _year == '2016preVFP':
          events['mtrigger'] = events.HLT.IsoMu24 | events.HLT.IsoTkMu24
          events['etrigger'] = events.HLT.Ele27_WPTight_Gsf
          mpt_threshold = (26, 15)
          ept_threshold = (20, 29)
        elif _year == '2016postVFP':
          events['mtrigger'] = events.HLT.IsoMu24 | events.HLT.IsoTkMu24
          events['etrigger'] = events.HLT.Ele27_WPTight_Gsf
          mpt_threshold = (26, 15)
          ept_threshold = (20, 29)
        elif _year == '2017':
          events['mtrigger'] = events.HLT.IsoMu27
          events['etrigger'] = events.HLT.Ele35_WPTight_Gsf
          mpt_threshold = (29, 15)
          ept_threshold = (20, 37)
        elif _year == '2018':
          events['mtrigger'] = events.HLT.IsoMu24
          events['etrigger'] = events.HLT.Ele32_WPTight_Gsf
          mpt_threshold = (26, 15)
          ept_threshold = (20, 34)

#        events['mpt_threshold'] = ak.where(events.mtrigger, numpy.repeat(mpt_threshold_val[0], len(events)), numpy.repeat(mpt_threshold_val[1], len(events)))
#        events['ept_threshold'] = ak.where(events.mtrigger, numpy.repeat(ept_threshold_val[0], len(events)), numpy.repeat(ept_threshold_val[1], len(events)))
    
        #Choose em channel and IsoMu Trigger
        emevents = events[(events.channel == 0) & (events.mtrigger|events.etrigger)]
        E_collections = emevents.Electron
        M_collections = emevents.Muon

        #Kinematics Selections
        E_collections.Target = ((abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.05) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
        M_collections.Target = ((abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.05) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))
        emevents["Electron", "Target_m"] = ((E_collections.pt > ept_threshold[0]) & E_collections.Target)
        emevents["Muon", "Target_m"] = ((M_collections.pt > mpt_threshold[0]) & (M_collections.Target))
        emevents["Electron", "Target_e"] = ((E_collections.pt > ept_threshold[1]) & E_collections.Target)
        emevents["Muon", "Target_e"] = ((M_collections.pt > mpt_threshold[1]) & (M_collections.Target))

        #Trig Matching
        M_collections = M_collections[emevents.Muon.Target_m==1]
        M_collections = ak.pad_none(M_collections, 1, axis=-1)
        #https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L60
        trg_collections = emevents.TrigObj[emevents.TrigObj.id == 13]
        mtrg_Match = (ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1) & emevents.mtrigger)
        mtrg_Match = ak.fill_none(mtrg_Match, False)
        #Trig Matching
        E_collections = E_collections[emevents.Electron.Target_e==1]
        E_collections = ak.pad_none(E_collections, 1, axis=-1)
        trg_collections = emevents.TrigObj[emevents.TrigObj.id == 11]
        etrg_Match = (ak.any((E_collections[:,0].delta_r(trg_collections) < 0.5),1) & emevents.etrigger)
        etrg_Match = ak.fill_none(etrg_Match, False)

        emevents["Electron", "Target"] = ((emevents.Electron.Target_m & mtrg_Match) | (emevents.Electron.Target_e & etrg_Match))
        emevents["Muon", "Target"] = ((emevents.Muon.Target_m & mtrg_Match) | (emevents.Muon.Target_e & etrg_Match))

        emevents['mtrigger'], emevents['etrigger'] = mtrg_Match, etrg_Match

        emevents = emevents[(emevents.mtrigger|emevents.etrigger)]

        E_collections = emevents.Electron[emevents.Electron.Target==1]
        M_collections = emevents.Muon[emevents.Muon.Target==1]

        #Opposite Charge
        E_charge = ak.fill_none(ak.pad_none(E_collections.charge, 1, axis=-1), 0, axis=-1)[:,0]
        M_charge = ak.fill_none(ak.pad_none(M_collections.charge, 1, axis=-1), 0, axis=-1)[:,0]
        opp_charge = E_charge*M_charge==-1
        same_charge = E_charge*M_charge==1
    
        emevents['opp_charge'] = opp_charge
        if sameCharge:
          emevents = emevents[opp_charge | same_charge]
        else:
          emevents = emevents[opp_charge]

        return emevents

    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        #make interesting variables
        if 'LFV' in emevents.metadata["dataset"]:
          if '125' in emevents.metadata["dataset"]:
            emevents["label"] = numpy.ones(len(emevents)) 
          else:
            mpoint = int(emevents.metadata["dataset"].split('_')[-1][1:4])
            emevents["label"] = numpy.repeat(mpoint, len(emevents)) 
        elif emevents.metadata["dataset"]=='SingleMuon' or emevents.metadata["dataset"] == 'data': 
          emevents["label"] = numpy.repeat(3, len(emevents))
        else:
          emevents["label"] = numpy.zeros(len(emevents))
        emevents["sample"] = numpy.repeat(self._samples.index(emevents.metadata["dataset"]), len(emevents))
        emevents["ept"] = Electron_collections.pt
        emevents["mpt"] = Muon_collections.pt
        return emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(self._year, events, sameCharge=True)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = Corrections(emevents, massrange=(115,135))
          if len(emevents)>0:
            SF_fun = SF(self._lumiWeight, self._year, self._btag_sf, self._m_sf, self._e_sf, self._evaluator)
            emevents = SF_fun.evaluate(emevents)
            emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)

            for var in self.var_ :
                out[var+'_0jets'].add( processor.column_accumulator( emevents[emevents.nJet30 == 0][var].to_numpy() ) )
                out[var+'_1jets'].add( processor.column_accumulator( emevents[emevents.nJet30 == 1][var].to_numpy() ) )
                out[var+'_2jets'].add( processor.column_accumulator( emevents[emevents.nJet30 >= 2][var].to_numpy() ) )
            for var in self.var_1jet_ :
                out[var+'_1jets'].add( processor.column_accumulator( emevents[emevents.nJet30 == 1][var].to_numpy() ) )
                out[var+'_2jets'].add( processor.column_accumulator( emevents[emevents.nJet30 >= 2][var].to_numpy() ) )

            for var in self.var_2jet_ :
                out[var+'_2jets'].add( processor.column_accumulator( emevents[emevents.nJet30 >= 2][var].to_numpy() ) )
 
        return out

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':
  current = os.path.dirname(os.path.realpath(__file__))
#  sys.path.append(os.path.dirname(current))
#  import find_samples
  years = ['2016preVFP', '2016postVFP', '2017', '2018']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
    if '2016' in year:
      QCDhist=["hist_em_qcd_osss_ss_corr hist_em_qcd_osss_ss_corr Corrections/QCD/em_qcd_osss_2016.root", "hist_em_qcd_osss_os_corr hist_em_qcd_osss_os_corr Corrections/QCD/em_qcd_osss_2016.root"]
      TrackerMu_Hi=["trackerMu_Hi NUM_TrackerMuons_DEN_genTracks/abseta_p_value Corrections/TrackerMu/2016HighPt.json"]
      if 'pre' in year:
        TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json"]
      else:
        TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json"]
         
    elif '2017' in year:
      QCDhist=["hist_em_qcd_osss_ss_corr hist_em_qcd_osss_ss_corr Corrections/QCD/em_qcd_osss_2017.root", "hist_em_qcd_osss_os_corr hist_em_qcd_osss_os_corr Corrections/QCD/em_qcd_osss_2017.root"]
      TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json"]
      TrackerMu_Hi=["trackerMu_Hi NUM_TrackerMuons_DEN_genTracks/abseta_p_value Corrections/TrackerMu/2017HighPt.json"]
    elif '2018' in year:
      QCDhist=["hist_em_qcd_osss_ss_corr hist_em_qcd_osss_closureOS Corrections/QCD/em_qcd_osss_2018.root", "hist_em_qcd_osss_os_corr hist_em_qcd_extrap_uncert Corrections/QCD/em_qcd_osss_2018.root"]
      TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json"]
      TrackerMu_Hi=["trackerMu_Hi NUM_TrackerMuons_DEN_genTracks/abseta_p_value Corrections/TrackerMu/2018HighPt.json"]

    btag_sf = correctionlib.CorrectionSet.from_file(f"jsonpog-integration/POG/BTV/{year}_UL/btagging.json.gz")
    muon_sf = correctionlib.CorrectionSet.from_file(f"jsonpog-integration/POG/MUO/{year}_UL/muon_Z.json.gz")
    electron_sf = correctionlib.CorrectionSet.from_file(f"jsonpog-integration/POG/EGM/{year}_UL/electron.json.gz")
    ext = extractor()
    ext.add_weight_sets(QCDhist)
    ext.add_weight_sets(TrackerMu)
    ext.add_weight_sets(TrackerMu_Hi)
    ext.finalize()
    evaluator = ext.make_evaluator()

    processor_instance = MyDF(lumiWeight, year, btag_sf, muon_sf, electron_sf, evaluator)#, *find_samples.samples_to_run['makeDF'])
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
