from coffea import processor, hist
from coffea.util import save
from coffea.lookup_tools import extractor
import awkward as ak
import numba, numpy, json, os, sys

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
    if jets==None:
        return (emVar).pt/(lep1.pt+lep2.pt)
    elif len(jets)==1:
        return (emVar + jets[0]).pt/(lep1.pt+lep2.pt+jets[0].pt)
    elif len(jets)==2:
        return (emVar + jets[0] +jets[1]).pt/(lep1.pt+lep2.pt+jets[0].pt+jets[1].pt)
    else:
        return -999
    
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

class MyDF(processor.ProcessorABC):
    def __init__(self, lumiWeight, year):
        self._samples = []
        self._lumiWeight = lumiWeight
        self._year = year
        self._accumulator = processor.dict_accumulator({})
        self.var_ = ['opp_charge', "is2016preVFP", "is2016postVFP", "is2017", "is2018", "sample", "label", "weight", "njets", "e_m_Mass", "met", "eEta", "mEta", "mpt_Per_e_m_Mass", "ept_Per_e_m_Mass", "empt", "emEta", "DeltaEta_e_m", "DeltaPhi_e_m", "DeltaR_e_m", "Rpt_0", "e_met_mT", "m_met_mT", "e_met_mT_Per_e_m_Mass", "m_met_mT_Per_e_m_Mass", "pZeta85", "pZeta15", "pZeta", "pZetaVis"]
        self.var_1jet_ = ["j1pt", "j1Eta", "DeltaEta_j1_em", "DeltaPhi_j1_em", "DeltaR_j1_em", "Zeppenfeld_1", "Rpt_1"]
        self.var_2jet_ = ["isVBFcat", "j2pt", "j2Eta", "j1_j2_mass", "DeltaEta_em_j1j2", "DeltaPhi_em_j1j2", "DeltaR_em_j1j2", "DeltaEta_j2_em", "DeltaPhi_j2_em", "DeltaR_j2_em", "DeltaEta_j1_j2", "DeltaPhi_j1_j2", "DeltaR_j1_j2", "Zeppenfeld", "Zeppenfeld_DeltaEta", "absZeppenfeld_DeltaEta", "cen", "Rpt", "pt_cen", "pt_cen_Deltapt", "abspt_cen_Deltapt", "Ht_had", "Ht"]
        for var in self.var_+self.var_1jet_+self.var_2jet_ :
            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))

    def sample_list(self, *argv):
        self._samples = argv

    @property
    def accumulator(self):
        return self._accumulator
    
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
        emevents["Electron", "Target"] = ((E_collections.pt > 24) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.045) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
        emevents["Muon", "Target"] = ((M_collections.pt > mpt_threshold) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.045) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

        E_collections = emevents.Electron[emevents.Electron.Target==1]
        M_collections = emevents.Muon[emevents.Muon.Target==1]

        #Opposite Charge
        E_charge = ak.fill_none(ak.pad_none(E_collections.charge, 1, axis=-1), 0, axis=-1)
        M_charge = ak.fill_none(ak.pad_none(M_collections.charge, 1, axis=-1), 0, axis=-1)
        opp_charge = ak.flatten(E_charge*M_charge==-1)
        same_charge = ak.flatten(E_charge*M_charge==1)

        emevents['opp_charge'] = opp_charge
        emevents = emevents[opp_charge | same_charge]

        #Trig Matching
        M_collections = emevents.Muon
        trg_collections = emevents.TrigObj

        M_collections = M_collections[M_collections.Target==1]
        #https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L60
        trg_collections = trg_collections[trg_collections.id == 13]

        trg_Match = ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1)

        return emevents[trg_Match & (emevents.nJet30>1)]

    def Corrections(self, emevents):
        Electron_collections = emevents.Electron[emevents.Electron.Target==1]
        Muon_collections = emevents.Muon[emevents.Muon.Target==1]
        MET_collections = emevents.MET
        Jet_collections = emevents.Jet[emevents.Jet.passJet30ID==1]

        #Jet corrections
        Jet_collections['pt'] = Jet_collections['pt_nom']
        Jet_collections['mass'] = Jet_collections['mass_nom']
        #padding to have at least "2 jets"
        Jet_collections = ak.pad_none(Jet_collections, 2, clip=True)

        #MET corrections
        if emevents.metadata["dataset"]!='SingleMuon' and emevents.metadata["dataset"]!='data':
            MET_collections['phi'] = MET_collections['T1Smear_phi'] 
            MET_collections['pt'] = MET_collections['T1Smear_pt'] 
        else:
            MET_collections['phi'] = MET_collections['T1_phi'] 
            MET_collections['pt'] = MET_collections['T1_pt'] 

        #MET corrections Electron
        Electron_collections['pt'] = Electron_collections['pt']/Electron_collections['eCorr']
        MET_collections = MET_collections+Electron_collections[:,0]
        Electron_collections['pt'] = Electron_collections['pt']*Electron_collections['eCorr']
        MET_collections = MET_collections-Electron_collections[:,0]
        
        #Muon pT corrections
        MET_collections = MET_collections+Muon_collections[:,0]
        Muon_collections['mass'] = Muon_collections['mass']*Muon_collections['corrected_pt']/Muon_collections['pt']
        Muon_collections['pt'] = Muon_collections['corrected_pt']
        MET_collections = MET_collections-Muon_collections[:,0]

        #ensure Jets are pT-ordered
        Jet_collections = Jet_collections[ak.argsort(Jet_collections.pt, axis=1, ascending=False)]
        #Take the first leptons
        Electron_collections = Electron_collections[:,0]
        Muon_collections = Muon_collections[:,0]
        emVar = Electron_collections + Muon_collections

        massRange = (emVar.mass<160) & (emVar.mass>110)
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	
    
    def SF(self, emevents):
        passDeepJet30 = (emevents.Jet.passJet30ID) & emevents.Jet.passDeepJet_L
        Muon_collections = emevents.Muon[emevents.Muon.Target==1][:,0]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1][:,0]
          
        if emevents.metadata["dataset"]=='SingleMuon' or emevents.metadata["dataset"] == 'data': 
          SF = ak.sum(passDeepJet30,-1)==2 #numpy.ones(len(emevents))
        else:
          #Get bTag SF
          nbtag = ak.sum(passDeepJet30,-1)
          bTagSF = numpy.zeros(len(nbtag))
          btagSF_deepjet_ = emevents.Jet.btagSF_deepjet_L*passDeepJet30
          btagSF_deepjet_=btagSF_deepjet_[btagSF_deepjet_!=0]
          bTagSF_fast(bTagSF, btagSF_deepjet_, nbtag)
          #bTag/PU/Gen Weights
          SF = bTagSF*emevents.puWeight*emevents.genWeight

          #PU/PF/Gen Weights
          if self._year != '2018':
            SF = SF*emevents.PrefireWeight
          #Zvtx
          if self._year == '2017':
            SF = 0.991*SF
 
          #Muon SF
          SF = SF*Muon_collections.Trigger_SF*Muon_collections.ID_SF*Muon_collections.ISO_SF
  
          #Electron SF and lumi
          SF = SF*Electron_collections.Reco_SF*Electron_collections.ID_SF*self._lumiWeight[emevents.metadata["dataset"]]
  
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
          QCDhist=["hist_em_qcd_osss_ss_corr hist_em_qcd_osss_ss_corr em_qcd_osss_2016.root", "hist_em_qcd_osss_os_corr hist_em_qcd_osss_os_corr em_qcd_osss_2016.root"]
          QCDexp="((njets==0)*(2.852125+-0.282871*dr)+(njets==1)*(2.792455+-0.295163*dr)+(njets>=2)*(2.577038+-0.290886*dr))*ss_corr*os_corr"
        elif '2017' in self._year:
          QCDhist=["hist_em_qcd_osss_ss_corr hist_em_qcd_osss_ss_corr em_qcd_osss_2017.root", "hist_em_qcd_osss_os_corr hist_em_qcd_osss_os_corr em_qcd_osss_2017.root"]
          QCDexp="((njets==0)*(3.221108+-0.374644*dr)+(njets==1)*(2.818298+-0.287438*dr)+(njets>=2)*(2.944477+-0.342411*dr))*ss_corr*os_corr"
        elif '2018' in self._year:
          QCDhist=["hist_em_qcd_osss_ss_corr hist_em_qcd_osss_closureOS em_qcd_osss_2018.root", "hist_em_qcd_osss_os_corr hist_em_qcd_extrap_uncert em_qcd_osss_2018.root"]
          QCDexp="((njets==0)*(2.042-0.05889**dr)+(njets==1)*(2.827-0.2907*dr)+(njets>=2)*(2.9-0.3641*dr))*ss_corr*os_corr"

        ext = extractor()
        ext.add_weight_sets(QCDhist)
        ext.finalize()
        evaluator = ext.make_evaluator()
        ss_corr, os_corr = evaluator["hist_em_qcd_osss_ss_corr"](mpt, ept), evaluator["hist_em_qcd_osss_os_corr"](mpt, ept)
        osss = ak.numexpr.evaluate(QCDexp) 

        emevents["weight"] = SF*ak.fill_none(osss, 1, axis=-1)

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
        
        return emevents

    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        #make interesting variables
        emevents["sample"] = numpy.repeat(self._samples.index(emevents.metadata["dataset"]), len(emevents))
        emevents["njets"] = emevents.nJet30
        emVar = Electron_collections + Muon_collections
        emevents["eEta"] = Electron_collections.eta
        emevents["mEta"] = Muon_collections.eta
        emevents["e_m_Mass"] = emVar.mass
        emevents["mpt_Per_e_m_Mass"] = Muon_collections.pt/emevents["e_m_Mass"]
        emevents["ept_Per_e_m_Mass"] = Electron_collections.pt/emevents["e_m_Mass"]
        emevents["empt"] = emVar.pt
        emevents["emEta"] = emVar.eta
        emevents["DeltaEta_e_m"] = abs(Muon_collections.eta - Electron_collections.eta)
        emevents["DeltaPhi_e_m"] = Muon_collections.delta_phi(Electron_collections)
        emevents["DeltaR_e_m"] = Muon_collections.delta_r(Electron_collections)
        emevents["Rpt_0"] = Rpt(Muon_collections, Electron_collections)

        emevents["met"] = MET_collections.pt

        emevents["e_met_mT"] = mT(Electron_collections, MET_collections)
        emevents["m_met_mT"] = mT(Muon_collections, MET_collections)
        emevents["e_met_mT_Per_e_m_Mass"] = emevents["e_met_mT"]/emevents["e_m_Mass"]
        emevents["m_met_mT_Per_e_m_Mass"] = emevents["m_met_mT"]/emevents["e_m_Mass"]

        pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
        emevents["pZeta85"] = pZeta_ - 0.85*pZetaVis_
        emevents["pZeta15"] = pZeta_ - 1.5*pZetaVis_
        emevents["pZeta"] = pZeta_
        emevents["pZetaVis"] = pZetaVis_

        emevents['j1pt'] = Jet_collections[:,0].pt
        emevents['j1Eta'] = Jet_collections[:,0].eta

        emevents["DeltaEta_j1_em"] = abs(Jet_collections[:,0].eta - emVar.eta)
        emevents["DeltaPhi_j1_em"] = Jet_collections[:,0].delta_phi(emVar)
        emevents["DeltaR_j1_em"] = Jet_collections[:,0].delta_r(emVar)

        emevents["Zeppenfeld_1"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0]])
        emevents["Rpt_1"] = Rpt(Muon_collections, Electron_collections, [Jet_collections[:,0]])

        emevents['j2pt'] = Jet_collections[:,1].pt
        emevents['j2Eta'] = Jet_collections[:,1].eta
        emevents["j1_j2_mass"] = (Jet_collections[:,0] + Jet_collections[:,1]).mass

        emevents["DeltaEta_em_j1j2"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - emVar.eta)
        emevents["DeltaPhi_em_j1j2"] = (Jet_collections[:,0] + Jet_collections[:,1]).delta_phi(emVar)
        emevents["DeltaR_em_j1j2"] = (Jet_collections[:,0] + Jet_collections[:,1]).delta_r(emVar)

        emevents["DeltaEta_j2_em"] = abs(Jet_collections[:,1].eta - emVar.eta)
        emevents["DeltaPhi_j2_em"] = Jet_collections[:,1].delta_phi(emVar)
        emevents["DeltaR_j2_em"] = Jet_collections[:,1].delta_r(emVar)

        emevents["DeltaEta_j1_j2"] = abs(Jet_collections[:,0].eta - Jet_collections[:,1].eta)

        emevents["isVBFcat"] = ((emevents["j1_j2_mass"] > 400) & (emevents["DeltaEta_j1_j2"] > 2.5)) 

        emevents["DeltaPhi_j1_j2"] = Jet_collections[:,0].delta_phi(Jet_collections[:,1])
        emevents["DeltaR_j1_j2"] = Jet_collections[:,0].delta_r(Jet_collections[:,1])

        emevents["Zeppenfeld"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
        emevents["Zeppenfeld_DeltaEta"] = emevents["Zeppenfeld"]/emevents["DeltaEta_j1_j2"]
        emevents["absZeppenfeld_DeltaEta"] = abs(emevents["Zeppenfeld_DeltaEta"])
        emevents["cen"] = numpy.exp(-4*emevents["Zeppenfeld_DeltaEta"]**2)

        emevents["Rpt"] = Rpt(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])

        emevents["pt_cen"] = pt_cen(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
        emevents["pt_cen_Deltapt"] = emevents["pt_cen"]/(Jet_collections[:,0] - Jet_collections[:,1]).pt
        emevents["abspt_cen_Deltapt"] = abs(emevents["pt_cen_Deltapt"])

        emevents["Ht_had"] = ak.sum(Jet_collections.pt, 1)
        emevents["Ht"] = ak.sum(Jet_collections.pt, 1) + Muon_collections.pt + Electron_collections.pt
        return emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        if len(emevents)>0:
          emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = self.Corrections(emevents)
          emevents = self.SF(emevents)
          emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)

          for var in self.var_+self.var_1jet_+self.var_2jet_ :
              out[var+'_2jets'].add( processor.column_accumulator( emevents[emevents.nJet30 == 2][var].to_numpy() ) )
 
        return out

    def postprocess(self, accumulator):
        return accumulator


if __name__ == '__main__':
#  current = os.path.dirname(os.path.realpath(__file__))
#  sys.path.append(os.path.dirname(current))
#  import find_samples
  years = ['2017', '2018']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
    processor_instance = MyDF(lumiWeight, year)#, *find_samples.samples_to_run['makeDF'])
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
