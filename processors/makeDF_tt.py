from coffea import processor, hist
from coffea.util import save
from coffea.lookup_tools import extractor
import awkward as ak
import numba, correctionlib, numpy, json, os, sys
from kinematics import *
from Vetos import *
from Corrections import *

@numba.njit
def bTagSF_fast(bTagSF, btagSF_deepjet, nbtag):
    for i in range(len(nbtag)):
        if (nbtag[i]) >= 2:
            acc_hi=0
            for j in range(len(btagSF_deepjet[i])-1):
                acc=btagSF_deepjet[i][j]
                for k in range(j+1, len(btagSF_deepjet[i])):
                    acc*=btagSF_deepjet[i][k]
                    for z in range(len(btagSF_deepjet[i])):
                        if z!=j and z!=k:
                            acc*=(1-btagSF_deepjet[i][z])
                    acc_hi+=acc
                    acc=btagSF_deepjet[i][j]
            bTagSF[i]=acc_hi

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
        self.var_ = ["opp_charge", "is2016preVFP", "is2016postVFP", "is2017", "is2018", "sample", "label", "weight", "njets", "e_m_Mass", "met", "eEta", "mEta", "mpt", "ept", "empt", "DeltaEta_e_m", "DeltaPhi_em_met"]
        self.var_1jet_ = ["j1pt", "j1Eta", "DeltaEta_j1_em"]
        self.var_2jet_ = ["isVBFcat", "j2pt", "j2Eta", "j1_j2_mass", "DeltaEta_em_j1j2", "DeltaEta_j1_j2", "Zeppenfeld_DeltaEta", "Rpt", "pt_cen_Deltapt", "Ht_had"]
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
    
    def SF(self, emevents):
        Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
          
        if emevents.metadata["dataset"]=='SingleMuon' or emevents.metadata["dataset"] == 'data': 
          SF = ak.sum(emevents.Jet.passDeepJet_L,1)==2 #numpy.ones(len(emevents))
        else:
          #Get bTag SF
          #old btag
          #btagSF_deepjet_L = self._btag_sf.eval("central", emevents.Jet.hadronFlavour, abs(emevents.Jet.eta), emevents.Jet.pt_nom)
          jet_flat = ak.flatten(emevents.Jet)
          btagSF_deepjet_L = numpy.zeros(len(jet_flat))
          jet_light = ak.where((jet_flat.passDeepJet_L) & (jet_flat.hadronFlavour==0))
          jet_heavy = ak.where((jet_flat.passDeepJet_L) & (jet_flat.hadronFlavour!=0))
          array_light = self._btag_sf["deepJet_incl"].evaluate("central", "L", jet_flat[jet_light].hadronFlavour.to_numpy(), abs(jet_flat[jet_light].eta).to_numpy(), jet_flat[jet_light].pt_nom.to_numpy())
          array_heavy = self._btag_sf["deepJet_comb"].evaluate("central", "L", jet_flat[jet_heavy].hadronFlavour.to_numpy(), abs(jet_flat[jet_heavy].eta).to_numpy(), jet_flat[jet_heavy].pt_nom.to_numpy())
          btagSF_deepjet_L[jet_light] = array_light
          btagSF_deepjet_L[jet_heavy] = array_heavy
          btagSF_deepjet_L = ak.unflatten(btagSF_deepjet_L, ak.num(emevents.Jet))
          nbtag = ak.sum(emevents.Jet.passDeepJet_L,-1)
          bTagSF = numpy.zeros(len(nbtag))
          bTagSF_fast(bTagSF, btagSF_deepjet_L, nbtag)
 
          #bTag/PU/Gen Weights
          SF = bTagSF*emevents.puWeight*emevents.genWeight

          #PU/PF/Gen Weights
          if self._year != '2018':
            SF = SF*emevents.L1PreFiringWeight.Nom
            #SF = SF*emevents.PrefireWeight
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
          MuTrigger_SF = self._m_sf[triggerstr].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 
          MuID_SF = self._m_sf["NUM_TightID_DEN_TrackerMuons"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 
          MuISO_SF = self._m_sf["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 

          SF = SF*MuTrigger_SF*MuID_SF*MuISO_SF*Trk_SF*Trk_SF_Hi
          #SF = SF*Muon_collections.Trigger_SF*Muon_collections.ID_SF*Muon_collections.ISO_SF*Trk_SF*Trk_SF_Hi
  
          #Electron SF and lumi
          EleReco_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","RecoAbove20", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
          EleIDnoISO_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","wp80noiso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
          SF = SF*EleReco_SF*EleIDnoISO_SF*self._lumiWeight[emevents.metadata["dataset"]]
#          SF = SF*Electron_collections.Reco_SF*Electron_collections.IDnoISO_SF*self._lumiWeight[emevents.metadata["dataset"]]
  
          SF = SF.to_numpy()
          SF[abs(SF)>10] = 0

        #osss factor for QCD
        emevents["njets"] = emevents.nJet30
        emevents["DeltaR_e_m"] = Muon_collections.delta_r(Electron_collections)
        mpt = ak.mask(Muon_collections['pt'], emevents['opp_charge']!=1)
        ept = ak.mask(Electron_collections['pt'], emevents['opp_charge']!=1)
        dr = ak.mask(emevents["DeltaR_e_m"], emevents['opp_charge']!=1)
        njets = ak.mask(emevents["njets"], emevents['opp_charge']!=1)

        if '2016' in self._year:
          QCDexp="((njets==0)*(2.852125+-0.282871*dr)+(njets==1)*(2.792455+-0.295163*dr)+(njets>=2)*(2.577038+-0.290886*dr))*ss_corr*os_corr"
        elif '2017' in self._year:
          QCDexp="((njets==0)*(3.221108+-0.374644*dr)+(njets==1)*(2.818298+-0.287438*dr)+(njets>=2)*(2.944477+-0.342411*dr))*ss_corr*os_corr"
        elif '2018' in self._year:
          QCDexp="((njets==0)*(2.042-0.05889**dr)+(njets==1)*(2.827-0.2907*dr)+(njets>=2)*(2.9-0.3641*dr))*ss_corr*os_corr"

        ss_corr, os_corr = self._evaluator["hist_em_qcd_osss_ss_corr"](mpt, ept), self._evaluator["hist_em_qcd_osss_os_corr"](mpt, ept)
        osss = ak.numexpr.evaluate(QCDexp) 

        emevents["weight"] = SF*ak.fill_none(osss, 1, axis=-1)
        return emevents

    def interesting(self, emevents):
        #make interesting variables
        emevents["is2016preVFP"] = numpy.ones(len(emevents)) if self._year == '2016preVFP' else numpy.zeros(len(emevents))
        emevents["is2016postVFP"] = numpy.ones(len(emevents)) if self._year == '2016postVFP' else numpy.zeros(len(emevents))
        emevents["is2017"] = numpy.ones(len(emevents)) if self._year == '2017' else numpy.zeros(len(emevents))
        emevents["is2018"] = numpy.ones(len(emevents)) if self._year == '2018' else numpy.zeros(len(emevents))

        if 'LFV' in emevents.metadata["dataset"]:
          if '125' in emevents.metadata["dataset"]:
            emevents["label"] = numpy.ones(len(emevents)) 
          elif '130' in emevents.metadata["dataset"]:
            emevents["label"] = numpy.repeat(130, len(emevents)) 
          else:
            emevents["label"] = numpy.repeat(120, len(emevents)) 
        elif emevents.metadata["dataset"]=='SingleMuon' or emevents.metadata["dataset"] == 'data': 
          emevents["label"] = numpy.repeat(3, len(emevents))
        else:
          emevents["label"] = numpy.zeros(len(emevents))
        emevents["sample"] = numpy.repeat(self._samples.index(emevents.metadata["dataset"]), len(emevents))
        
        return emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = Vetos(self._year, events, sameCharge=True)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = Corrections(emevents)
          emevents = self.SF(emevents)
          emevents = interestingKin(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
          emevents = self.interesting(emevents)

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
        #old btag
        #btag_sf = BTagScaleFactor("Corrections/bTag/DeepJet_106XUL16SF.csv", "LOOSE")
      else:
        TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json"]
        #old btag
        #btag_sf = BTagScaleFactor("Corrections/bTag/DeepJet_106XUL16SF.csv", "LOOSE")
         
    elif '2017' in year:
      QCDhist=["hist_em_qcd_osss_ss_corr hist_em_qcd_osss_ss_corr Corrections/QCD/em_qcd_osss_2017.root", "hist_em_qcd_osss_os_corr hist_em_qcd_osss_os_corr Corrections/QCD/em_qcd_osss_2017.root"]
      TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json"]
      TrackerMu_Hi=["trackerMu_Hi NUM_TrackerMuons_DEN_genTracks/abseta_p_value Corrections/TrackerMu/2017HighPt.json"]
      #old btag
      #btag_sf = BTagScaleFactor("Corrections/bTag/DeepCSV_106XUL17SF_WPonly_V2p1.csv", "LOOSE")
    elif '2018' in year:
      QCDhist=["hist_em_qcd_osss_ss_corr hist_em_qcd_osss_closureOS Corrections/QCD/em_qcd_osss_2018.root", "hist_em_qcd_osss_os_corr hist_em_qcd_extrap_uncert Corrections/QCD/em_qcd_osss_2018.root"]
      TrackerMu=["trackerMu NUM_TrackerMuons_DEN_genTracks/abseta_pt_value Corrections/TrackerMu/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json"]
      TrackerMu_Hi=["trackerMu_Hi NUM_TrackerMuons_DEN_genTracks/abseta_p_value Corrections/TrackerMu/2018HighPt.json"]
      #old btag
      #btag_sf = BTagScaleFactor("Corrections/bTag/DeepJet_106XUL18SF_WPonly_V1p1.csv", "LOOSE")

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
