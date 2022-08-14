import awkward as ak
import numpy
class SF:
    def __init__(self, lumiWeight, year, btag_sf, muon_sf, electron_sf, electron_sf_pri, evaluator):
        self._lumiWeight = lumiWeight
        self._year = year
        self._btag_sf = btag_sf
        self._e_sf = electron_sf
        self._e_sf_pri = electron_sf_pri
        self._m_sf = muon_sf
        self._evaluator = evaluator
        self._jecYear = self._year[:4]
    
    def evaluate(self, emevents, doQCD=False, doSys=False):
        Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
          
        if 'data' in emevents.metadata["dataset"]: 
          SF = ak.sum(emevents.Jet.passDeepJet_L,1)==0
        else:
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
    
          #bTag/PU/Gen Weights
          SF = bTagSF*emevents.puWeight*emevents.genWeight
    
          #PU/PF/Gen Weights
          if self._year != '2018':
            SF = SF*emevents.L1PreFiringWeight.Nom
            #SF = SF*emevents.PrefireWeight
          #Zvtx
          #if self._year == '2017':
          #  SF = ak.where(emevents['etrigger'], 0.991, 1)*SF
    
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
          
          MuTrigger_SF = numpy.ones(len(emevents))
          Mu_pass = ak.where(emevents['mtrigger'])
          MuTrigger_SF[Mu_pass] = self._m_sf[triggerstr].evaluate(f"{self._year}_UL", abs(Muon_collections[Mu_pass].eta).to_numpy(), Muon_collections[Mu_pass].pt.to_numpy(), "sf")
          MuID_SF = self._m_sf["NUM_TightID_DEN_TrackerMuons"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 
          MuISO_SF = self._m_sf["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "sf") 
    
          SF = SF*MuTrigger_SF*MuID_SF*MuISO_SF*Trk_SF*Trk_SF_Hi
    
          #Electron SF and lumi
          EleReco_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","RecoAbove20", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
          #EleIDISO_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","wp80iso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
          EleIDnoISO_SF = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sf","wp80noiso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
          EleISO_SF = self._e_sf_pri["UL-Electron-ID-SF"].evaluate(self._year,"sf","Iso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
          Ele_pass = ak.where(emevents['etrigger'])
          EleTrig_SF = numpy.ones(len(emevents))
     
          EleTrig_SF[Ele_pass] = self._e_sf_pri["UL-Electron-ID-SF"].evaluate(self._year,"sf","Trig", Electron_collections[Ele_pass].eta.to_numpy(), Electron_collections[Ele_pass].pt.to_numpy()) 

          #Zvtx
          if self._year == '2017':
            EleTrig_SF[Ele_pass]*=0.991

          SF = SF*EleISO_SF*EleTrig_SF*EleReco_SF*EleIDnoISO_SF*self._lumiWeight[emevents.metadata["dataset"]]
    
          SF = SF.to_numpy()
          SF[abs(SF)>10] = 0
    
          if doSys:
            MuTrigger_SF_Up = numpy.ones(len(emevents))
            MuTrigger_SF_Up[Mu_pass] = self._m_sf[triggerstr].evaluate(f"{self._year}_UL", abs(Muon_collections[Mu_pass].eta).to_numpy(), Muon_collections[Mu_pass].pt.to_numpy(), "systup")
            MuID_SF_Up = self._m_sf["NUM_TightID_DEN_TrackerMuons"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "systup") 
            MuISO_SF_Up = self._m_sf["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "systup") 
            MuTrigger_SF_Down = numpy.ones(len(emevents))
            MuTrigger_SF_Down[Mu_pass] = self._m_sf[triggerstr].evaluate(f"{self._year}_UL", abs(Muon_collections[Mu_pass].eta).to_numpy(), Muon_collections[Mu_pass].pt.to_numpy(), "systdown")
            MuID_SF_Down = self._m_sf["NUM_TightID_DEN_TrackerMuons"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "systdown") 
            MuISO_SF_Down = self._m_sf["NUM_TightRelIso_DEN_TightIDandIPCut"].evaluate(f"{self._year}_UL", abs(Muon_collections.eta).to_numpy(), Muon_collections.pt.to_numpy(), "systdown") 
            emevents[f"weight_mID_Up"] = SF*MuID_SF_Up/MuID_SF
            emevents[f"weight_mIso_Up"] = SF*MuISO_SF_Up/MuISO_SF
            emevents[f"weight_mTrg_Up"] = SF*MuTrigger_SF_Up/MuTrigger_SF
            emevents[f"weight_mID_Down"] = SF*MuID_SF_Down/MuID_SF
            emevents[f"weight_mIso_Down"] = SF*MuISO_SF_Down/MuISO_SF
            emevents[f"weight_mTrg_Down"] = SF*MuTrigger_SF_Down/MuTrigger_SF
            EleReco_SF_Up = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sfup","RecoAbove20", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
            EleIDnoISO_SF_Up = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sfup","wp80noiso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
            EleISO_SF_Up = self._e_sf_pri["UL-Electron-ID-SF"].evaluate(self._year,"sfup","Iso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
            EleTrig_SF_Up = numpy.ones(len(emevents))
            EleTrig_SF_Up[Ele_pass] = self._e_sf_pri["UL-Electron-ID-SF"].evaluate(self._year,"sfup","Trig", Electron_collections[Ele_pass].eta.to_numpy(), Electron_collections[Ele_pass].pt.to_numpy()) 
            EleReco_SF_Down = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sfdown","RecoAbove20", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
            EleIDnoISO_SF_Down = self._e_sf["UL-Electron-ID-SF"].evaluate(self._year,"sfdown","wp80noiso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
            EleISO_SF_Down = self._e_sf_pri["UL-Electron-ID-SF"].evaluate(self._year,"sfdown","Iso", Electron_collections.eta.to_numpy(), Electron_collections.pt.to_numpy())
            EleTrig_SF_Down = numpy.ones(len(emevents))
            EleTrig_SF_Down[Ele_pass] = self._e_sf_pri["UL-Electron-ID-SF"].evaluate(self._year,"sfdown","Trig", Electron_collections[Ele_pass].eta.to_numpy(), Electron_collections[Ele_pass].pt.to_numpy()) 
            if self._year == '2017':
              EleTrig_SF_Up[Ele_pass]*=0.991
              EleTrig_SF_Down[Ele_pass]*=0.991
            emevents[f"weight_eReco_Up"] = SF*EleReco_SF_Up/EleReco_SF
            emevents[f"weight_eID_Up"] = SF*EleIDnoISO_SF_Up/EleIDnoISO_SF
            emevents[f"weight_eIso_Up"] = SF*EleISO_SF_Up/EleISO_SF
            emevents[f"weight_eTrig_Up"] = SF*EleTrig_SF_Up/EleTrig_SF
            emevents[f"weight_eReco_Down"] = SF*EleReco_SF_Down/EleReco_SF
            emevents[f"weight_eID_Down"] = SF*EleIDnoISO_SF_Down/EleIDnoISO_SF
            emevents[f"weight_eIso_Down"] = SF*EleISO_SF_Down/EleISO_SF
            emevents[f"weight_eTrig_Down"] = SF*EleTrig_SF_Down/EleTrig_SF
    
            for other_year in ['2016', '2017', '2018']:
              emevents[f"weight_bTag_{other_year}_Up"] = SF
              emevents[f"weight_bTag_{other_year}_Down"] = SF
              emevents[f"weight_pu_{other_year}_Up"] = SF
              emevents[f"weight_pu_{other_year}_Down"] = SF
            for other_year in ['2016', '2017']:
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
            emevents[f"weight_bTag_{self._jecYear}_Up"] = SF*bTagSF_Up/bTagSF
            emevents[f"weight_bTag_{self._jecYear}_Down"] = SF*bTagSF_Down/bTagSF
    
            #PU Up/Down 
            emevents[f"weight_pu_{self._jecYear}_Up"] = SF*emevents.puWeightUp/emevents.puWeight
            emevents[f"weight_pu_{self._jecYear}_Down"] = SF*emevents.puWeightDown/emevents.puWeight
    
            #Pre-firing Up/Down
            if self._year != '2018':
              emevents[f"weight_pf_{self._jecYear}_Up"] = SF*emevents.L1PreFiringWeight.Up/emevents.L1PreFiringWeight.Nom
              emevents[f"weight_pf_{self._jecYear}_Down"] = SF*emevents.L1PreFiringWeight.Dn/emevents.L1PreFiringWeight.Nom
    
            #Scale uncertainty
            emevents[f"weight_scalep5p5"] = SF*emevents.LHEScaleWeight[:,0]
            emevents[f"weight_scale22"] = SF*emevents.LHEScaleWeight[:,8]
    
            #PDF and alpha_s
            #https://lhapdfsets.web.cern.ch/current/NNPDF31_nnlo_as_0118_mc_hessian_pdfas/NNPDF31_nnlo_as_0118_mc_hessian_pdfas.info
            #SF_theory[0] ... etc
            weight_lhe = numpy.einsum("ij,i->ij", emevents.LHEPdfWeight.to_numpy(), SF)
            for i in range(103):
              emevents[f"weight_lhe{i}"] = weight_lhe[:,i] 
    
        if doQCD:
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
    
        else:    
          emevents["weight"] = SF
    
        return emevents

def Corrections(emevents, massrange=(90,180)):
    Electron_collections = emevents.Electron[emevents.Electron.Target==1]
    Muon_collections = emevents.Muon[emevents.Muon.Target==1]
    MET_collections = emevents.MET
    Jet_collections = emevents.Jet[emevents.Jet.passJet30ID==1]
    #Jet corrections
    Jet_collections['pt'] = Jet_collections['pt_nom']
    Jet_collections['mass'] = Jet_collections['mass_nom']

    #MET corrections
    MET_collections['phi'] = MET_collections['T1_phi'] 
    MET_collections['pt'] = MET_collections['T1_pt'] 

    ##MET corrections Electron
    #Electron_collections['pt'] = Electron_collections['pt']/Electron_collections['eCorr']
    #MET_collections = MET_collections+Electron_collections[:,0]
    #Electron_collections['pt'] = Electron_collections['pt']*Electron_collections['eCorr']
    #MET_collections = MET_collections-Electron_collections[:,0]
    
    #Muon pT corrections
    #MET_collections = MET_collections+Muon_collections[:,0]
    Muon_collections['pt'] = Muon_collections['corrected_pt']
    #MET_collections = MET_collections-Muon_collections[:,0]

    #ensure Jets are pT-ordered
    Jet_collections = Jet_collections[ak.argsort(Jet_collections.pt, axis=1, ascending=False)]
    #padding to have at least "2 jets"
    Jet_collections = ak.pad_none(Jet_collections, 2)

    #Take the first leptons
    Electron_collections = Electron_collections[:,0]
    Muon_collections = Muon_collections[:,0]
    emVar = Electron_collections + Muon_collections

    #if emevents.metadata["dataset"] == 'SingleMuon' or emevents.metadata["dataset"] == 'data':
    #    massRange = ((emVar.mass<115) & (emVar.mass>110)) | ((emVar.mass<160) & (emVar.mass>135))
    #else:
    massRange = (emVar.mass<massrange[1]) & (emVar.mass>massrange[0])
    return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	

