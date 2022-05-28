from coffea import processor, hist
from coffea.util import save
from coffea.lookup_tools import extractor
#old btag
#from coffea.btag_tools import BTagScaleFactor
import awkward as ak
import correctionlib, numpy, json, os, sys
from kinematics import *
from Vetos import *
from Corrections import *

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
        self.var_ = ["opp_charge", "is2016preVFP", "is2016postVFP", "is2017", "is2018", "sample", "label", "weight", "njets", "e_m_Mass", "e_m_Mass_reso", "met", "eEta", "mEta", "mpt", "ept", "empt", "emEta", "DeltaEta_e_m", "DeltaR_e_m", "DeltaPhi_e_met", "DeltaPhi_m_met", "DeltaPhi_em_met", "e_mvaTTH", "m_mvaTTH", "e_mvaFall17V2Iso", "e_mvaFall17V2noIso", "mtrigger", "etrigger"]
        self.var_1jet_ = ["j1pt", "j1Eta", "DeltaEta_j1_em", "DeltaR_j1_em"]
        self.var_2jet_ = ["isVBFcat", "j2pt", "j2Eta", "j1_j2_mass", "DeltaEta_em_j1j2", "DeltaR_em_j1j2", "DeltaEta_j2_em", "DeltaR_j2_em", "DeltaEta_j1_j2", "DeltaR_j1_j2", "Zeppenfeld_DeltaEta", "Rpt", "pt_cen_Deltapt", "Ht_had"]
        for var in self.var_+self.var_1jet_+self.var_2jet_:
            self._accumulator[var] = processor.column_accumulator(numpy.array([]))
#            self._accumulator[var+'_0jets'] = processor.column_accumulator(numpy.array([]))
#            self._accumulator[var+'_1jets'] = processor.column_accumulator(numpy.array([]))
#            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))
#        for var in self.var_1jet_ :
#            self._accumulator[var+'_1jets'] = processor.column_accumulator(numpy.array([]))
#            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))
#        for var in self.var_2jet_ :
#            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))

    def sample_list(self, *argv):
        self._samples = argv

    @property
    def accumulator(self):
        return self._accumulator

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
        emevents["is2016preVFP"] = numpy.ones(len(emevents)) if self._year == '2016preVFP' else numpy.zeros(len(emevents))
        emevents["is2016postVFP"] = numpy.ones(len(emevents)) if self._year == '2016postVFP' else numpy.zeros(len(emevents))
        emevents["is2017"] = numpy.ones(len(emevents)) if self._year == '2017' else numpy.zeros(len(emevents))
        emevents["is2018"] = numpy.ones(len(emevents)) if self._year == '2018' else numpy.zeros(len(emevents))
        emevents["sample"] = numpy.repeat(self._samples.index(emevents.metadata["dataset"]), len(emevents))

        tmpElectron_collections = ak.zip(
          {
            "pt":   Electron_collections.pt,
            "eta":  Electron_collections.eta,
            "phi":  Electron_collections.phi,
            "mass": Electron_collections.mass
          },
          with_name="PtEtaPhiMLorentzVector",
        )
        tmpElectron_collections = tmpElectron_collections*(tmpElectron_collections.energy + Electron_collections.energyErr)/tmpElectron_collections.energy

        tmpMuon_collections = ak.zip(
          {
            "pt":   Muon_collections.pt,
            "eta":  Muon_collections.eta,
            "phi":  Muon_collections.phi,
            "mass": Muon_collections.mass
          },
          with_name="PtEtaPhiMLorentzVector",
        )
        tmpMuon_collections['pt'] = tmpMuon_collections.pt + Muon_collections.ptErr
        e_m_Mass_ereso = (Muon_collections + tmpElectron_collections).mass - emevents["e_m_Mass"]
        e_m_Mass_mreso = (tmpMuon_collections + Electron_collections).mass - emevents["e_m_Mass"]

        emevents["e_m_Mass_reso"] = numpy.sqrt( e_m_Mass_mreso**2+e_m_Mass_ereso**2 )

        emevents["m_mvaTTH"] = Muon_collections.mvaTTH
        emevents["e_mvaTTH"] = Electron_collections.mvaTTH
        emevents["e_mvaFall17V2noIso"] = Electron_collections.mvaFall17V2noIso
        emevents["e_mvaFall17V2Iso"] = Electron_collections.mvaFall17V2Iso

        emVar = Electron_collections + Muon_collections
        emevents["emEta"] = emVar.eta
        emevents["DeltaR_e_m"] = Muon_collections.delta_r(Electron_collections)

#        emevents["ePhi"] = Electron_collections.phi
#        emevents["mPhi"] = Muon_collections.phi
#        emevents["eIso"] = Electron_collections.pfRelIso03_all
#        emevents["mIso"] = Muon_collections.pfRelIso04_all
#        emevents["emPhi"] = emVar.phi
#        emevents["DeltaPhi_e_m"] = Muon_collections.delta_phi(Electron_collections)
#        emevents["Rpt_0"] = Rpt(Muon_collections, Electron_collections)

#        emevents["metPhi"] = MET_collections.phi

        #emevents["e_met_mT"] = mT(MET_collections, Electron_collections)
        #emevents["m_met_mT"] = mT(MET_collections, Muon_collections)
        #emevents["e_m_met_mT"] = mT3(MET_collections, Electron_collections, Muon_collections)
        #emevents["e_met_mT_Per_e_m_Mass"] = emevents["e_met_mT"]/emevents["e_m_Mass"]
        #emevents["m_met_mT_Per_e_m_Mass"] = emevents["m_met_mT"]/emevents["e_m_Mass"]
        #emevents["e_m_met_mT_Per_e_m_Mass"] = emevents["e_m_met_mT"]/emevents["e_m_Mass"]
        emevents["DeltaPhi_e_met"] = Electron_collections.delta_phi(MET_collections)
        emevents["DeltaPhi_m_met"] = Muon_collections.delta_phi(MET_collections)

        #pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
        #emevents["pZeta85"] = pZeta_ - 0.85*pZetaVis_
        #emevents["pZeta15"] = pZeta_ - 1.5*pZetaVis_
        #emevents["pZeta"] = pZeta_
        #emevents["pZetaVis"] = pZetaVis_

#        emevents["DeltaPhi_j1_em"] = Jet_collections[:,0].delta_phi(emVar)
        emevents["DeltaR_j1_em"] = Jet_collections[:,0].delta_r(emVar)

#        emevents["Zeppenfeld_1"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0]])
#        emevents["Rpt_1"] = Rpt(Muon_collections, Electron_collections, [Jet_collections[:,0]])
#        emevents["j1btagDeepFlavB"] = Jet_collections[:,0].btagDeepFlavB

        emevents['j2Eta'] = Jet_collections[:,1].eta

#        emevents["DeltaPhi_em_j1j2"] = (Jet_collections[:,0] + Jet_collections[:,1]).delta_phi(emVar)
        emevents["DeltaR_em_j1j2"] = (Jet_collections[:,0] + Jet_collections[:,1]).delta_r(emVar)

        emevents["DeltaEta_j2_em"] = abs(Jet_collections[:,1].eta - emVar.eta)
#        emevents["DeltaPhi_j2_em"] = Jet_collections[:,1].delta_phi(emVar)
        emevents["DeltaR_j2_em"] = Jet_collections[:,1].delta_r(emVar)

#        emevents["DeltaPhi_j1_j2"] = Jet_collections[:,0].delta_phi(Jet_collections[:,1])
        emevents["DeltaR_j1_j2"] = Jet_collections[:,0].delta_r(Jet_collections[:,1])

#        emevents["Zeppenfeld"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
#        emevents["absZeppenfeld_DeltaEta"] = abs(emevents["Zeppenfeld_DeltaEta"])
#        emevents["cen"] = numpy.exp(-4*emevents["Zeppenfeld_DeltaEta"]**2)

#        emevents["pt_cen"] = pt_cen(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
#        emevents["abspt_cen_Deltapt"] = abs(emevents["pt_cen_Deltapt"])
#        emevents["Ht"] = ak.sum(Jet_collections.pt, 1) + Muon_collections.pt + Electron_collections.pt

        #emevents["j2btagDeepFlavB"] = Jet_collections[:,1].btagDeepFlavB
        return emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = Vetos(self._year, events, sameCharge=True)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = Corrections(emevents, (110,160))
          SF_fun = SF(self._lumiWeight, self._year, self._btag_sf, self._m_sf, self._e_sf, self._evaluator)
          emevents = SF_fun.evaluate(emevents, doQCD=True)
          emevents = interestingKin(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
          emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)

          for var in self.var_+self.var_1jet_+self.var_2jet_ :
            new_array = emevents[var].to_numpy()
            if type(new_array)==numpy.ma.core.MaskedArray:
              new_array = new_array.filled(numpy.nan)
            out[var].add( processor.column_accumulator( new_array ) )

#              out[var+'_0jets'].add( processor.column_accumulator( emevents[emevents.nJet30 == 0][var].to_numpy() ) )
#              out[var+'_1jets'].add( processor.column_accumulator( emevents[emevents.nJet30 == 1][var].to_numpy() ) )
#              out[var+'_2jets'].add( processor.column_accumulator( emevents[emevents.nJet30 >= 2][var].to_numpy() ) )
#          for var in self.var_1jet_ :
#              out[var+'_1jets'].add( processor.column_accumulator( emevents[emevents.nJet30 == 1][var].to_numpy() ) )
#              out[var+'_2jets'].add( processor.column_accumulator( emevents[emevents.nJet30 >= 2][var].to_numpy() ) )
#
#          for var in self.var_2jet_ :
#              out[var+'_2jets'].add( processor.column_accumulator( emevents[emevents.nJet30 >= 2][var].to_numpy() ) )
 
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
