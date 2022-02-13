from coffea import processor, hist
from coffea.util import save
from coffea.lookup_tools import extractor
from coffea.btag_tools import BTagScaleFactor
import xgboost as xgb
import pandas as pd
import awkward as ak
import correctionlib, numpy, json, os

def pZeta(leg1, leg2, MET_px, MET_py):
    leg1x = numpy.cos(leg1.phi)
    leg2x = numpy.cos(leg2.phi)
    leg1y = numpy.sin(leg1.phi)
    leg2y = numpy.sin(leg2.phi)
    zetaX = leg1x + leg2x
    zetaY = leg1y + leg2y
    zetaR = numpy.sqrt(zetaX*zetaX + zetaY*zetaY)
    
    zetaX = numpy.where((zetaR > 0.), zetaX/zetaR, zetaX)
    zetaY = numpy.where((zetaR > 0.), zetaY/zetaR, zetaY)
    
    visPx = leg1.px + leg2.px
    visPy = leg1.py + leg2.py
    pZetaVis = visPx*zetaX + visPy*zetaY
    px = visPx + MET_px
    py = visPy + MET_py
    
    pZeta = px*zetaX + py*zetaY
    
    return (pZeta, pZetaVis)

def Rpt(lep1, lep2, jets=None):
    emVar = lep1+lep2
    return (emVar + jets[0] +jets[1]).pt/(lep1.pt+lep2.pt+jets[0].pt+jets[1].pt)
   # if jets==None:
   #     return (emVar).pt/(lep1.pt+lep2.pt)
   # elif len(jets)==1:
   #     return (emVar + jets[0]).pt/(lep1.pt+lep2.pt+jets[0].pt)
   # elif len(jets)==2:
   #     return (emVar + jets[0] +jets[1]).pt/(lep1.pt+lep2.pt+jets[0].pt+jets[1].pt)
   # else:
   #     return -999
    
def Zeppenfeld(lep1, lep2, jets):
    emVar = lep1+lep2
    if len(jets)==1:
        return emVar.eta - (jets[0].eta)/2
    elif len(jets)==2:
        return emVar.eta - (jets[0].eta + jets[1].eta)/2
    else:
        return -999

def mT(lep, met):
    return numpy.sqrt(abs((numpy.sqrt(lep.mass**2+lep.pt**2) + met.pt)**2 - (lep+met).pt**2))

def pt_cen(lep1, lep2, jets):
    emVar = lep1+lep2
    if len(jets)==1:
        return emVar.pt - jets[0].pt/2
    elif len(jets)==2:
        return emVar.pt - (jets[0] + jets[1]).pt/2
    else:
        return -999

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
        self.sfUnc = sum([[f'pu_{year}', f'bTag_{year}'] for year in ['2017', '2018', '2016preVFP', '2016postVFP']], [])
        self.sfUnc += ['pf_2016preVFP', 'pf_2016postVFP', 'pf_2017']
        self.theoUnc = [f'lhe{i}' for i in range(103)] + ['scalep5p5', 'scale22']
        self.leptonUnc = ['me']#['ees', 'eer', 'me']
        self._accumulator = processor.dict_accumulator({})
        self._accumulator[f'e_m_Mass'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'isVBFcat'] = processor.column_accumulator(numpy.bool_([]))
        self._accumulator[f'isVBF'] = processor.column_accumulator(numpy.bool_([]))
        self._accumulator[f'isHerwig'] = processor.column_accumulator(numpy.bool_([]))
        self._accumulator[f'njets'] = processor.column_accumulator(numpy.ubyte([]))
        self._accumulator[f'mva'] = processor.column_accumulator(numpy.array([]))
        self._accumulator[f'year'] = processor.column_accumulator(numpy.ubyte([]))
        self._accumulator[f'weight'] = processor.column_accumulator(numpy.array([]))
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
    
    def BDTscore(self, XFrame, isVBF=False):
        if isVBF:
           model_load = self._BDTmodels["model_VBF"]
        else:
           model_load = self._BDTmodels[f"model_GG"]
        return model_load.predict_proba(XFrame)

    def Vetos(self, events):
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

        #Choose em channel and IsoMu Trigger
        emevents = events[(events.channel == 0) & (trigger == 1)]

        E_collections = emevents.Electron
        M_collections = emevents.Muon

        #Kinematics Selections
        emevents["Electron", "Target"] = ((E_collections.pt > 24) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.05) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
        emevents["Muon", "Target"] = ((M_collections.pt > mpt_threshold) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.05) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

        E_collections = emevents.Electron[emevents.Electron.Target==1]
        M_collections = emevents.Muon[emevents.Muon.Target==1]

        #Opposite Charge
        E_charge = ak.fill_none(ak.pad_none(E_collections.charge, 1, axis=-1), 0, axis=-1)[:,0]
        M_charge = ak.fill_none(ak.pad_none(M_collections.charge, 1, axis=-1), 0, axis=-1)[:,0]
        opp_charge = E_charge*M_charge==-1
        emevents = emevents[opp_charge]

        #Trig Matching
        M_collections = emevents.Muon
        trg_collections = emevents.TrigObj

        M_collections = M_collections[M_collections.Target==1]
        #https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L60
        trg_collections = trg_collections[trg_collections.id == 13]

        trg_Match = ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1)

        return emevents[trg_Match]
   
    def Corrections(self, emevents):
        Electron_collections = emevents.Electron[emevents.Electron.Target==1]
        Muon_collections = emevents.Muon[emevents.Muon.Target==1]
        MET_collections = emevents.MET
        Jet_collections = emevents.Jet[emevents.Jet.passJet30ID==1]

        #Jet corrections
        Jet_collections['pt'] = Jet_collections['pt_nom']
        Jet_collections['mass'] = Jet_collections['mass_nom']

        #MET corrections
        if emevents.metadata["dataset"]!='SingleMuon' and emevents.metadata["dataset"]!='data':
            MET_collections['phi'] = MET_collections['T1Smear_phi'] 
            MET_collections['pt'] = MET_collections['T1Smear_pt'] 
        else:
            MET_collections['phi'] = MET_collections['T1_phi'] 
            MET_collections['pt'] = MET_collections['T1_pt'] 

        #Muon pT corrections
        Muon_collections['mass'] = Muon_collections['mass']*Muon_collections['corrected_pt']/Muon_collections['pt']
        Muon_collections['pt'] = Muon_collections['corrected_pt']

        #ensure Jets are pT-ordered
        Jet_collections = Jet_collections[ak.argsort(Jet_collections.pt, axis=1, ascending=False)]
        #padding to have at least "2 jets"
        Jet_collections = ak.pad_none(Jet_collections, 2, clip=True)

        #Take the first leptons
        Electron_collections = Electron_collections[:,0]
        Muon_collections = Muon_collections[:,0]
        emVar = Electron_collections + Muon_collections

        massRange = (emVar.mass<160) & (emVar.mass>110)
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	
    
    def SF(self, emevents):
        #Get bTag SF
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

        SF = bTagSF*emevents.puWeight*emevents.genWeight

        #PU/PF/Gen Weights
        if self._year != '2018':
          SF = SF*emevents.L1PreFiringWeight.Nom

        Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
        
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

        #Electron SF and lumi
        EleReco_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","RecoAbove20", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
        EleIDnoISO_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","wp80noiso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
        SF = SF*EleReco_SF*EleIDnoISO_SF*self._lumiWeight[emevents.metadata["dataset"]]
        SF = SF.to_numpy()
        SF[abs(SF)>10] = 0
        emevents["weight"] = SF
        
        for other_year in ['2016preVFP', '2016postVFP', '2017', '2018']:
          emevents[f"weight_bTag_{other_year}_Up"] = SF
          emevents[f"weight_bTag_{other_year}_Down"] = SF
          emevents[f"weight_pu_{other_year}_Up"] = SF
          emevents[f"weight_pu_{other_year}_Down"] = SF
        for other_year in ['2016preVFP', '2016postVFP', '2017']:
          emevents[f"weight_pf_{other_year}_Up"] = SF
          emevents[f"weight_pf_{other_year}_Down"] = SF

	#bTag Up/Down
        array_light_up = self._btag_sf["deepJet_incl"].evaluate("up_uncorrelated", "L", jet_flat[jet_light].hadronFlavour.to_numpy(), abs(jet_flat[jet_light].eta).to_numpy(), jet_flat[jet_light].pt_nom.to_numpy())
        array_heavy_up = self._btag_sf["deepJet_comb"].evaluate("up_uncorrelated", "L", jet_flat[jet_heavy].hadronFlavour.to_numpy(), abs(jet_flat[jet_heavy].eta).to_numpy(), jet_flat[jet_heavy].pt_nom.to_numpy())
        array_light_down = self._btag_sf["deepJet_incl"].evaluate("down_uncorrelated", "L", jet_flat[jet_light].hadronFlavour.to_numpy(), abs(jet_flat[jet_light].eta).to_numpy(), jet_flat[jet_light].pt_nom.to_numpy())
        array_heavy_down = self._btag_sf["deepJet_comb"].evaluate("down_uncorrelated", "L", jet_flat[jet_heavy].hadronFlavour.to_numpy(), abs(jet_flat[jet_heavy].eta).to_numpy(), jet_flat[jet_heavy].pt_nom.to_numpy())

        btagSF_deepjet_L_down = numpy.zeros(len(jet_flat))
        btagSF_deepjet_L_up = numpy.zeros(len(jet_flat))

        btagSF_deepjet_L_down[jet_heavy] = array_heavy_down
        btagSF_deepjet_L_down[jet_light] = array_light_down

        btagSF_deepjet_L_up[jet_heavy] = array_heavy_up
        btagSF_deepjet_L_up[jet_light] = array_light_up

        btagSF_deepjet_L_down = ak.unflatten(btagSF_deepjet_L_down, ak.num(emevents.Jet))
        btagSF_deepjet_L_up = ak.unflatten(btagSF_deepjet_L_up, ak.num(emevents.Jet))

        bTagSF_Down = ak.prod(1-btagSF_deepjet_L_down, axis=1)
        bTagSF_Up = ak.prod(1-btagSF_deepjet_L_up, axis=1)
        emevents[f"weight_bTag_{self._year}_Up"] = SF*bTagSF_Up/bTagSF
        emevents[f"weight_bTag_{self._year}_Down"] = SF*bTagSF_Down/bTagSF

        #PU Up/Down 
        emevents[f"weight_pu_{self._year}_Up"] = SF*emevents.puWeightUp/emevents.puWeight
        emevents[f"weight_pu_{self._year}_Down"] = SF*emevents.puWeightDown/emevents.puWeight

	#Pre-firing Up/Down
        if self._year != '2018':
          emevents[f"weight_pf_{self._year}_Up"] = SF*emevents.L1PreFiringWeight.Up/emevents.L1PreFiringWeight.Nom
          emevents[f"weight_pf_{self._year}_Down"] = SF*emevents.L1PreFiringWeight.Dn/emevents.L1PreFiringWeight.Nom

	#Scale uncertainty
        emevents[f"weight_scalep5p5"] = SF*emevents.LHEScaleWeight[:,0]
        emevents[f"weight_scale22"] = SF*emevents.LHEScaleWeight[:,8]

	#PDF and alpha_s
	#https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_as_0118_mc_hessian_pdfas/NNPDF31_nnlo_as_0118_mc_hessian_pdfas.info
	#SF_theory[0] ... etc
        weight_lhe = numpy.einsum("ij,i->ij", emevents.LHEPdfWeight.to_numpy(), SF)
        for i in range(103):
          emevents[f"weight_lhe{i}"] = weight_lhe[:,i] 
        return emevents 

    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        #make interesting variables
        #zero/any no. of jets
        if '2016' in self._year:
          emevents["year"] = numpy.full(len(emevents), 0, dtype=numpy.ubyte)
        elif '2017' in self._year:
          emevents["year"] = numpy.full(len(emevents), 1, dtype=numpy.ubyte)
        else:
          emevents["year"] = numpy.full(len(emevents), 2, dtype=numpy.ubyte)
        if ('VBF' in emevents.metadata["dataset"]) and (not 'herwig' in emevents.metadata["dataset"]):
          emevents["isVBF"] = numpy.full(len(emevents), True, dtype=numpy.bool_) 
        else:
          emevents["isVBF"] = numpy.full(len(emevents), False, dtype=numpy.bool_) 
        if 'herwig' in emevents.metadata["dataset"]:
          emevents["isHerwig"] = numpy.full(len(emevents), True, dtype=numpy.bool_) 
        else:
          emevents["isHerwig"] = numpy.full(len(emevents), False, dtype=numpy.bool_)
        emVar = Electron_collections + Muon_collections
        emevents["e_m_Mass"] = emVar.mass
#        emevents["mpt_Per_e_m_Mass"] = Muon_collections.pt/emVar.mass
#        emevents["ept_Per_e_m_Mass"] = Electron_collections.pt/emVar.mass
        emevents["empt"] = emVar.pt
        emevents["DeltaEta_e_m"] = abs(Muon_collections.eta - Electron_collections.eta)
        emevents["met"] = MET_collections.pt
        emevents["DeltaPhi_em_met"] = emVar.delta_phi(MET_collections)
#        emevents["e_met_mT"] = mT(Electron_collections, MET_collections)
#        emevents["m_met_mT"] = mT(Muon_collections, MET_collections)

#        pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
#        emevents["pZeta85"] = pZeta_ - 0.85*pZetaVis_
        emevents["njets"] = emevents.nJet30 

        #1 jet
        emevents['j1pt'] = Jet_collections[:,0].pt
        emevents['j1Eta'] = Jet_collections[:,0].eta
        emevents["DeltaEta_j1_em"] = abs(Jet_collections[:,0].eta - emVar.eta)

        #2 or more jets
        emevents['j2pt'] = Jet_collections[:,1].pt
        emevents["j1_j2_mass"] = (Jet_collections[:,0] + Jet_collections[:,1]).mass
        emevents["DeltaEta_em_j1j2"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - emVar.eta)
        emevents["DeltaEta_j1_j2"] = abs(Jet_collections[:,0].eta - Jet_collections[:,1].eta)
        emevents["isVBFcat"] = ((emevents["njets"] >= 2) & (emevents["j1_j2_mass"] > 400) & (emevents["DeltaEta_j1_j2"] > 2.5)) 
        emevents["isVBFcat"] = ak.fill_none(emevents["isVBFcat"], False)
        emevents["Zeppenfeld_DeltaEta"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/emevents["DeltaEta_j1_j2"]
        emevents["Rpt"] = Rpt(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
        emevents["pt_cen_Deltapt"] = pt_cen(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/(Jet_collections[:,0] - Jet_collections[:,1]).pt
        emevents["Ht_had"] = ak.sum(Jet_collections.pt, 1)

        #Systematics

        for UpDown in ['Up', 'Down']:
#          #Electron Energy Scale
#          tmpElectron_collections = ak.copy(Electron_collections)
#          tmpElectron_collections['pt'] = tmpElectron_collections['pt']*tmpElectron_collections[f'dEscale{UpDown}']/Electron_collections['eCorr']
#          #Redo all Electron var
#  
#          #Electron Energy Reso
#          tmpElectron_collections = ak.copy(Electron_collections)
#          tmpElectron_collections['pt'] = tmpElectron_collections['pt']*tmpElectron_collections[f'dEsigma{UpDown}']/Electron_collections['eCorr']
#          #Redo all Electron var
  
  
          #Muon Energy Scale + Reso 
          tmpMuon_collections = ak.copy(Muon_collections)
          tmpMuon_collections['mass'] = tmpMuon_collections['mass']*tmpMuon_collections[f'corrected{UpDown}_pt']/tmpMuon_collections['pt']
          tmpMuon_collections['pt'] = tmpMuon_collections[f'corrected{UpDown}_pt']
          #Redo all Muon var
          tmpemVar = Electron_collections + tmpMuon_collections
          emevents[f'e_m_Mass_me_{UpDown}'] = tmpemVar.mass
#          emevents[f"mpt_Per_e_m_Mass_me_{UpDown}"] = tmpMuon_collections.pt/emevents[f'e_m_Mass_me_{UpDown}']
#          emevents[f"ept_Per_e_m_Mass_me_{UpDown}"] = Electron_collections.pt/emevents[f'e_m_Mass_me_{UpDown}']
          emevents[f"empt_me_{UpDown}"] = tmpemVar.pt
          emevents[f"DeltaEta_e_m_me_{UpDown}"] = abs(tmpMuon_collections.eta - Electron_collections.eta)
#          emevents[f"m_met_mT_me_{UpDown}"] = mT(tmpMuon_collections, MET_collections)
#          pZeta_, pZetaVis_ = pZeta(tmpMuon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
#          emevents[f"pZeta85_me_{UpDown}"] = pZeta_ - 0.85*pZetaVis_
          emevents["DeltaPhi_em_met_me_{UpDown}"] = tmpemVar.delta_phi(MET_collections)
          emevents[f"DeltaEta_j1_em_me_{UpDown}"] = abs(Jet_collections[:,0].eta - tmpemVar.eta)
          emevents[f"DeltaEta_em_j1j2_me_{UpDown}"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - tmpemVar.eta)
          emevents[f"Zeppenfeld_DeltaEta_me_{UpDown}"] = Zeppenfeld(tmpMuon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/emevents["DeltaEta_j1_j2"]
          emevents[f"Rpt_me_{UpDown}"] = Rpt(tmpMuon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
          emevents[f"pt_cen_Deltapt_me_{UpDown}"] = pt_cen(tmpMuon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/(Jet_collections[:,0] - Jet_collections[:,1]).pt

          #Unclustered
          tmpMET_collections = ak.copy(MET_collections)
          tmpMET_collections['phi'] = emevents.MET[f'T1Smear_phi_unclustEn{UpDown}'] 
          tmpMET_collections['pt'] = emevents.MET[f'T1Smear_pt_unclustEn{UpDown}'] 
  
          #Redo all MET var
          emevents[f"met_UnclusteredEn_{UpDown}"] = tmpMET_collections.pt
          emevents["DeltaPhi_em_met_UnclusteredEn_{UpDown}"] = emVar.delta_phi(tmpMET_collections)
#          emevents[f"e_met_mT_UnclusteredEn_{UpDown}"] = mT(Electron_collections, tmpMET_collections)
#          emevents[f"m_met_mT_UnclusteredEn_{UpDown}"] = mT(Muon_collections, tmpMET_collections)
#          pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  tmpMET_collections.px,  tmpMET_collections.py)
#          emevents["pZeta85_UnclusteredEn_{UpDown}"] = pZeta_ - 0.85*pZetaVis_
  
          #Jet Unc
          semitmpJet_collections = ak.copy(emevents.Jet)
          for jetUnc in self.jetUnc+self.jetyearUnc:
             if jetUnc in self.jetyearUnc and not self._jecYear in jetUnc:
               #ignore this year
               emevents[f"isVBFcat_{jetUnc}_{UpDown}"] = emevents["isVBFcat"] 
               emevents[f"njets_{jetUnc}_{UpDown}"] = emevents["njets"]
             else:
               if 'jer' in jetUnc: 
                 jetUncNoyear='jer'
               else:
                 jetUncNoyear=jetUnc
               semitmpJet_collections[f'passJet30ID_{jetUnc}{UpDown}'] = (Electron_collections.delta_r(semitmpJet_collections) > 0.4) & (Muon_collections.delta_r(semitmpJet_collections) > 0.4) & (semitmpJet_collections[f'pt_{jetUncNoyear}{UpDown}']>30) & ((semitmpJet_collections.jetId>>1) & 1) & (abs(semitmpJet_collections.eta)<4.7) & (((semitmpJet_collections.puId>>2)&1) | (semitmpJet_collections[f'pt_{jetUncNoyear}{UpDown}']>50))  
               tmpJet_collections = semitmpJet_collections[semitmpJet_collections[f'passJet30ID_{jetUnc}{UpDown}']==1]
               emevents[f"njets_{jetUnc}_{UpDown}"] = ak.num(tmpJet_collections)
               tmpJet_collections['pt'] = tmpJet_collections[f'pt_{jetUncNoyear}{UpDown}']
               tmpJet_collections['mass'] = tmpJet_collections[f'mass_{jetUncNoyear}{UpDown}']
               #ensure Jets are pT-ordered
               tmpJet_collections = tmpJet_collections[ak.argsort(tmpJet_collections.pt, axis=1, ascending=False)]
               #padding to have at least "2 jets"
               tmpJet_collections = ak.pad_none(tmpJet_collections, 2, clip=True)
               #MET
               tmpMET_collections = ak.copy(emevents.MET)
               tmpMET_collections['pt'] = emevents.MET[f'T1Smear_pt_{jetUncNoyear}{UpDown}']
               tmpMET_collections['phi'] = emevents.MET[f'T1Smear_phi_{jetUncNoyear}{UpDown}']
               #Redo all Jet/MET var
               emevents[f"met_{jetUnc}_{UpDown}"] = tmpMET_collections.pt
#               emevents[f"e_met_mT_{jetUnc}_{UpDown}"] = mT(Electron_collections, tmpMET_collections)
#               emevents[f"m_met_mT_{jetUnc}_{UpDown}"] = mT(Muon_collections, tmpMET_collections)
#               pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  tmpMET_collections.px,  tmpMET_collections.py)
#               emevents[f"pZeta85_{jetUnc}_{UpDown}"] = pZeta_ - 0.85*pZetaVis_
               emevents["DeltaPhi_em_met_{jetUnc}_{UpDown}"] = emVar.delta_phi(tmpMET_collections)
               emevents[f'j1pt_{jetUnc}_{UpDown}'] = tmpJet_collections[:,0].pt
               emevents[f'j1Eta_{jetUnc}_{UpDown}'] = tmpJet_collections[:,0].eta
               emevents[f"DeltaEta_j1_em_{jetUnc}_{UpDown}"] = abs(tmpJet_collections[:,0].eta - emVar.eta)
               emevents[f'j2pt_{jetUnc}_{UpDown}'] = tmpJet_collections[:,1].pt
               emevents[f"j1_j2_mass_{jetUnc}_{UpDown}"] = (tmpJet_collections[:,0] + tmpJet_collections[:,1]).mass
               emevents[f"DeltaEta_em_j1j2_{jetUnc}_{UpDown}"] = abs((tmpJet_collections[:,0] + tmpJet_collections[:,1]).eta - emVar.eta)
               emevents[f"DeltaEta_j1_j2_{jetUnc}_{UpDown}"] = abs(tmpJet_collections[:,0].eta - tmpJet_collections[:,1].eta)
               emevents[f"isVBFcat_{jetUnc}_{UpDown}"] = ((emevents[f"njets_{jetUnc}_{UpDown}"] >= 2) & (emevents[f"j1_j2_mass_{jetUnc}_{UpDown}"] > 400) & (emevents[f"DeltaEta_j1_j2_{jetUnc}_{UpDown}"] > 2.5)) 
               emevents[f"isVBFcat_{jetUnc}_{UpDown}"] = ak.fill_none(emevents[f"isVBFcat_{jetUnc}_{UpDown}"], False)
               emevents[f"Zeppenfeld_DeltaEta_{jetUnc}_{UpDown}"] = Zeppenfeld(Muon_collections, Electron_collections, [tmpJet_collections[:,0], tmpJet_collections[:,1]])/emevents[f"DeltaEta_j1_j2_{jetUnc}_{UpDown}"]
               emevents[f"Rpt_{jetUnc}_{UpDown}"] = Rpt(Muon_collections, Electron_collections, [tmpJet_collections[:,0], tmpJet_collections[:,1]])
               emevents[f"pt_cen_Deltapt_{jetUnc}_{UpDown}"] = pt_cen(Muon_collections, Electron_collections, [tmpJet_collections[:,0], tmpJet_collections[:,1]])/(tmpJet_collections[:,0] - tmpJet_collections[:,1]).pt
               emevents[f"Ht_had_{jetUnc}_{UpDown}"] = ak.sum(tmpJet_collections.pt, 1)

        return emevents

    def pandasDF(self, emevents, unc=None, UpDown=None, isJetSys=False):
        if unc==None:
          Xframe_GG = ak.to_pandas(emevents[self.var_GG_]).fillna(value=numpy.nan)
          Xframe_2jet_VBF = ak.to_pandas(emevents[self.var_2jet_VBF_])
 
          emevents_GG_nom = self.BDTscore(Xframe_GG)[:,1] 
          emevents_2jet_VBF_nom = self.BDTscore(Xframe_2jet_VBF, True)[:,1] 
          emevents['mva'] = (emevents.isVBFcat==0) * emevents_GG_nom + ((emevents.njets>=2) & (emevents.isVBFcat==1))* emevents_2jet_VBF_nom
        else:
          var_GG_alt = [i+f'_{unc}_{UpDown}' if i+f'_{unc}_{UpDown}' in emevents.fields else i for i in self.var_GG_] 
          var_2jet_VBF_alt = [i+f'_{unc}_{UpDown}' if i+f'_{unc}_{UpDown}' in emevents.fields else i for i in self.var_2jet_VBF_] 

          Xframe_GG = ak.to_pandas(emevents[var_GG_alt]).fillna(value=numpy.nan)
          Xframe_2jet_VBF = ak.to_pandas(emevents[var_2jet_VBF_alt])
  
          rename2_GG_ = {i:i.replace(f'_{unc}_{UpDown}', '') for i in var_GG_alt}
          rename2_VBF_ = {i:i.replace(f'_{unc}_{UpDown}', '') for i in var_2jet_VBF_alt}

          Xframe_GG.rename(columns=rename2_GG_, inplace = True)
          Xframe_2jet_VBF.rename(columns=rename2_VBF_, inplace = True)
     
          emevents_GG = self.BDTscore(Xframe_GG)[:,1] 
          emevents_2jet_VBF = self.BDTscore(Xframe_2jet_VBF, True)[:,1] 
          if isJetSys:
            emevents[f'mva_{unc}_{UpDown}'] = (emevents[f"isVBFcat_{unc}_{UpDown}"]==0) * emevents_GG + ((emevents[f"njets_{unc}_{UpDown}"]>=2) & (emevents[f"isVBFcat_{unc}_{UpDown}"]==1))* emevents_2jet_VBF
          else:
            emevents[f'mva_{unc}_{UpDown}'] = (emevents.isVBFcat==0) * emevents_GG + ((emevents.njets>=2) & (emevents.isVBFcat==1))* emevents_2jet_VBF
        return emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = self.Corrections(emevents)
          emevents = self.SF(emevents)
          emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
          emevents = self.pandasDF(emevents)
          for sys in self.leptonUnc+self.metUnc:
            emevents = self.pandasDF(emevents, sys, 'Up')
            emevents = self.pandasDF(emevents, sys, 'Down')

          for sys in self.jetUnc+self.jetyearUnc:
            emevents = self.pandasDF(emevents, sys, 'Up', True)
            emevents = self.pandasDF(emevents, sys, 'Down', True)

          for sys_var_ in out:
            if '_hist' in sys_var_:
              continue
            else:
              acc = emevents[sys_var_].to_numpy()
              out[sys_var_].add( processor.column_accumulator( acc ) )

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
    VBF_quan = numpy.load(f"XGBoost-for-HtoEMu/results/VBFcat_quantiles",allow_pickle=True)
    GG_quan = numpy.load(f"XGBoost-for-HtoEMu/results/GGcat_quantiles",allow_pickle=True)
    processor_instance = MyEMuPeak(lumiWeight, BDTmodels, BDTvars, year, btag_sf, muon_sf, electron_sf, evaluator, VBF_quan, GG_quan)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
