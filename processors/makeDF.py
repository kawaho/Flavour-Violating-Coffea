from coffea import processor, hist
from coffea.util import save
import awkward as ak
import numpy, json, os
def pZeta(leg1, leg2, MET_px, MET_py):
    leg1x = numpy.cos(leg1.phi)
    leg2x = numpy.cos(leg2.phi)
    leg1y = numpy.sin(leg1.phi)
    leg2y = numpy.sin(leg2.phi)
    zetaX = leg1x + leg2x
    zetaY = leg1y + leg2y
    zetaR = numpy.sqrt(zetaX*zetaX + zetaY*zetaY)
    
    numpy.where((zetaR > 0.), zetaX/zetaR, zetaX)
    numpy.where((zetaR > 0.), zetaY/zetaR, zetaY)
    zetaX = zetaX/zetaR
    zetaY = zetaY/zetaR
    
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
    def __init__(self, lumiWeight):
        self._lumiWeight = lumiWeight
        self._accumulator = processor.dict_accumulator({})
        self.var_ = ["label", "weight", "met", "eEta", "mEta", "mpt_Per_e_m_Mass", "ept_Per_e_m_Mass", "empt", "emEta", "DeltaEta_e_m", "DeltaPhi_e_m", "DeltaR_e_m", "Rpt_0", "e_met_mT", "m_met_mT", "e_met_mT_Per_e_m_Mass", "m_met_mT_Per_e_m_Mass", "pZeta85", "pZeta15", "pZeta", "pZetaVis"]
        self.var_1jet_ = ["j1pt", "j1Eta", "DeltaEta_j1_em", "DeltaPhi_j1_em", "DeltaR_j1_em", "Zeppenfeld_1", "Rpt_1"]
        self.var_2jet_ = ["j2pt", "j2Eta", "j1_j2_mass", "DeltaEta_em_j1j2", "DeltaPhi_em_j1j2", "DeltaR_em_j1j2", "DeltaEta_j2_em", "DeltaPhi_j2_em", "DeltaR_j2_em", "DeltaEta_j1_j2", "DeltaPhi_j1_j2", "DeltaR_j1_j2", "Zeppenfeld", "Zeppenfeld_DeltaEta", "absZeppenfeld_DeltaEta", "cen", "Rpt", "pt_cen", "pt_cen_Deltapt", "abspt_cen_Deltapt", "Ht_had", "Ht"]
        for var in self.var_ :
            self._accumulator[var+'_0jets'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[var+'_1jets'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))
        for var in self.var_1jet_ :
            self._accumulator[var+'_1jets'] = processor.column_accumulator(numpy.array([]))
            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))
        for var in self.var_2jet_ :
            self._accumulator[var+'_2jets'] = processor.column_accumulator(numpy.array([]))
    @property
    def accumulator(self):
        return self._accumulator
    
    def Vetos(self, events):
        #Choice em channel and Iso27
        emevents = events[(events.channel == 0) & (events.HLT.IsoMu27 == 1)]

        E_collections = emevents.Electron
        M_collections = emevents.Muon

        #Kinematics Selections
        emevents["Electron", "Target"] = ((E_collections.pt > 24) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.045) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1))
        emevents["Muon", "Target"] = ((M_collections.pt > 29) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.045) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

        E_collections = emevents.Electron[emevents.Electron.Target==1]
        M_collections = emevents.Muon[emevents.Muon.Target==1]

        #Opposite Charge
        E_charge = ak.fill_none(ak.pad_none(E_collections.charge, 1), 0)
        M_charge = ak.fill_none(ak.pad_none(M_collections.charge, 1), 0)
        opp_charge = ak.flatten(E_charge*M_charge==-1)

        emevents = emevents[opp_charge]

        #Trig Matching
        M_collections = emevents.Muon
        trg_collections = emevents.TrigObj

        M_collections = M_collections[M_collections.Target==1]
        trg_collections = trg_collections[(((trg_collections.filterBits >> 1) & 1)==1) & (trg_collections.id == 13) & (trg_collections.pt > 29) & (ak.num(M_collections) == 1)]

        trg_Match = ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1)

        return emevents[trg_Match]

    def Corrections(self, emevents):
        Electron_collections = emevents.Electron[emevents.Electron.Target==1]
        Muon_collections = emevents.Muon[emevents.Muon.Target==1]
        MET_collections = emevents.MET
        Jet_collections = emevents.Jet[emevents.Jet.passJet30ID==1]

        #ensure Jets are pT-ordered
        #Jet corrections
        Jet_collections['pt'] = Jet_collections['pt_nom']
        Jet_collections['mass'] = Jet_collections['mass_nom']

        if emevents.metadata["dataset"]!='data':
            #MET pT corrections
            MET_collections['phi'] = MET_collections['T1Smear_phi'] 
            MET_collections['pt'] = MET_collections['T1Smear_pt'] \
                                    - ak.flatten(Muon_collections['pt']) + ak.flatten(Muon_collections['corrected_pt'])\
                                    - ak.flatten(Electron_collections['pt']/Electron_collections['eCorr'])\
                                    + ak.flatten(Electron_collections['pt'])
        else:
            MET_collections['phi'] = MET_collections['T1_phi'] 
            MET_collections['pt'] = MET_collections['T1_pt'] \
                                    - ak.flatten(Muon_collections['pt']) + ak.flatten(Muon_collections['corrected_pt'])\
                                    - ak.flatten(Electron_collections['pt']/Electron_collections['eCorr'])\
                                    + ak.flatten(Electron_collections['pt'])


        #Muon pT corrections
        Muon_collections['pt'] = Muon_collections['corrected_pt']
        
        #ensure Jets are pT-ordered
        Jet_collections = Jet_collections[ak.argsort(Jet_collections.pt, axis=1, ascending=False)]

        #Take the first leptons
        Electron_collections = Electron_collections[:,0]
        Muon_collections = Muon_collections[:,0]
        emVar = Electron_collections + Muon_collections
        if emevents.metadata["dataset"]=='data':
            massRange = ((emVar.mass<115) & (emVar.mass>110)) | ((emVar.mass<160) & (emVar.mass>135))
        else:
            massRange = (emVar.mass<160) & (emVar.mass>110)
        
        return emevents[massRange], Electron_collections[massRange], Muon_collections[massRange], MET_collections[massRange], Jet_collections[massRange]	
    
    def SF(self, emevents):
        if emevents.metadata["dataset"]=='data': return numpy.ones(len(emevents))
        #Get bTag SF
        bTagSF_L = ak.prod(1-emevents.Jet.btagSF_deepjet_L*emevents.Jet.passDeepJet_L, axis=1)
        bTagSF_M = ak.prod(1-emevents.Jet.btagSF_deepjet_M*emevents.Jet.passDeepJet_M, axis=1)

        #PU/PF/Gen Weights
        SF = emevents.puWeight*emevents.PrefireWeight*emevents.genWeight

        Muon_collections = emevents.Muon[emevents.Muon.Target==1]
        Electron_collections = emevents.Electron[emevents.Electron.Target==1]
        
        #Muon SF
        SF = SF*Muon_collections.Trigger_SF*Muon_collections.ID_SF*Muon_collections.ISO_SF

        #Electron SF
        SF = SF*Electron_collections.Reco_SF*Electron_collections.ID_SF
        
        emevents["weight"] = ak.flatten(SF)*self._lumiWeight[emevents.metadata["dataset"]]
        emevents["label"] = numpy.ones(len(emevents), dtype=bool) if 'LFV' in emevents.metadata["dataset"] else numpy.zeros(len(emevents), dtype=bool)
        
        return emevents

    def interesting(self, emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections):
        #make interesting variables
        #zero/any no. of jets
        emVar = Electron_collections + Muon_collections
        emevents["eEta"] = Electron_collections.eta
        emevents["mEta"] = Muon_collections.eta
        emevents["mpt_Per_e_m_Mass"] = Muon_collections.pt/emVar.mass
        emevents["ept_Per_e_m_Mass"] = Electron_collections.pt/emVar.mass
        emevents["empt"] = emVar.pt
        emevents["emEta"] = emVar.eta
        emevents["DeltaEta_e_m"] = abs(Muon_collections.eta - Electron_collections.eta)
        emevents["DeltaPhi_e_m"] = Muon_collections.delta_phi(Electron_collections)
        emevents["DeltaR_e_m"] = Muon_collections.delta_r(Electron_collections)
        emevents["Rpt_0"] = Rpt(Muon_collections, Electron_collections)

        emevents["met"] = MET_collections.pt

        emevents["e_met_mT"] = mT(Electron_collections, MET_collections)
        emevents["m_met_mT"] = mT(Muon_collections, MET_collections)
        emevents["e_met_mT_Per_e_m_Mass"] = emevents["e_met_mT"]/emVar.mass
        emevents["m_met_mT_Per_e_m_Mass"] = emevents["m_met_mT"]/emVar.mass

        pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
        emevents["pZeta85"] = pZeta_ - 0.85*pZetaVis_
        emevents["pZeta15"] = pZeta_ - 1.5*pZetaVis_
        emevents["pZeta"] = pZeta_
        emevents["pZetaVis"] = pZetaVis_

        #one jets
        onejets_emevents = emevents[emevents.nJet30 >= 1]
        Electron_collections_1jet = Electron_collections[emevents.nJet30 >= 1]
        Muon_collections_1jet = Muon_collections[emevents.nJet30 >= 1]
        emVar_1jet = Electron_collections_1jet + Muon_collections_1jet

        onejets_emevents['j1pt'] = Jet_collections[emevents.nJet30 >= 1][:,0].pt
        onejets_emevents['j1Eta'] = Jet_collections[emevents.nJet30 >= 1][:,0].eta

        onejets_emevents["DeltaEta_j1_em"] = abs(Jet_collections[emevents.nJet30 >= 1][:,0].eta - emVar_1jet.eta)
        onejets_emevents["DeltaPhi_j1_em"] = Jet_collections[emevents.nJet30 >= 1][:,0].delta_phi(emVar_1jet)
        onejets_emevents["DeltaR_j1_em"] = Jet_collections[emevents.nJet30 >= 1][:,0].delta_r(emVar_1jet)

        onejets_emevents["Zeppenfeld_1"] = Zeppenfeld(Muon_collections_1jet, Electron_collections_1jet, [Jet_collections[emevents.nJet30 >= 1][:,0]])
        onejets_emevents["Rpt_1"] = Rpt(Muon_collections_1jet, Electron_collections_1jet, [Jet_collections[emevents.nJet30 >= 1][:,0]])

        #2 or more jets
        Multijets_emevents = emevents[emevents.nJet30 >= 2]

        Electron_collections_2jet = Electron_collections[emevents.nJet30 >= 2]
        Muon_collections_2jet = Muon_collections[emevents.nJet30 >= 2]
        emVar_2jet = Electron_collections_2jet + Muon_collections_2jet

        MET_collections_2jet = MET_collections[emevents.nJet30 >= 2]

        Multijets_emevents['j2pt'] = Jet_collections[emevents.nJet30 >= 2][:,1].pt
        Multijets_emevents['j2Eta'] = Jet_collections[emevents.nJet30 >= 2][:,1].eta
        Multijets_emevents["j1_j2_mass"] = (Jet_collections[emevents.nJet30 >= 2][:,0] + Jet_collections[emevents.nJet30 >= 2][:,1]).mass

        Multijets_emevents["DeltaEta_em_j1j2"] = abs((Jet_collections[emevents.nJet30 >= 2][:,0] + Jet_collections[emevents.nJet30 >= 2][:,1]).eta - emVar_2jet.eta)
        Multijets_emevents["DeltaPhi_em_j1j2"] = (Jet_collections[emevents.nJet30 >= 2][:,0] + Jet_collections[emevents.nJet30 >= 2][:,1]).delta_phi(emVar_2jet)
        Multijets_emevents["DeltaR_em_j1j2"] = (Jet_collections[emevents.nJet30 >= 2][:,0] + Jet_collections[emevents.nJet30 >= 2][:,1]).delta_r(emVar_2jet)

        Multijets_emevents["DeltaEta_j1_em"] = abs(Jet_collections[emevents.nJet30 >= 2][:,0].eta - emVar_2jet.eta)
        Multijets_emevents["DeltaPhi_j1_em"] = Jet_collections[emevents.nJet30 >= 2][:,0].delta_phi(emVar_2jet)
        Multijets_emevents["DeltaR_j1_em"] = Jet_collections[emevents.nJet30 >= 2][:,0].delta_r(emVar_2jet)
        Multijets_emevents["DeltaEta_j2_em"] = abs(Jet_collections[emevents.nJet30 >= 2][:,1].eta - emVar_2jet.eta)
        Multijets_emevents["DeltaPhi_j2_em"] = Jet_collections[emevents.nJet30 >= 2][:,1].delta_phi(emVar_2jet)
        Multijets_emevents["DeltaR_j2_em"] = Jet_collections[emevents.nJet30 >= 2][:,1].delta_r(emVar_2jet)

        Multijets_emevents["DeltaEta_j1_j2"] = abs(Jet_collections[emevents.nJet30 >= 2][:,0].eta - Jet_collections[emevents.nJet30 >= 2][:,1].eta)
        Multijets_emevents["DeltaPhi_j1_j2"] = Jet_collections[emevents.nJet30 >= 2][:,0].delta_phi(Jet_collections[emevents.nJet30 >= 2][:,1])
        Multijets_emevents["DeltaR_j1_j2"] = Jet_collections[emevents.nJet30 >= 2][:,0].delta_r(Jet_collections[emevents.nJet30 >= 2][:,1])

        Multijets_emevents["Zeppenfeld"] = Zeppenfeld(Muon_collections_2jet, Electron_collections_2jet, [Jet_collections[emevents.nJet30 >= 2][:,0], Jet_collections[emevents.nJet30 >= 2][:,1]])
        Multijets_emevents["Zeppenfeld_DeltaEta"] = Multijets_emevents["Zeppenfeld"]/Multijets_emevents["DeltaEta_j1_j2"]
        Multijets_emevents["absZeppenfeld_DeltaEta"] = abs(Multijets_emevents["Zeppenfeld_DeltaEta"])
        Multijets_emevents["cen"] = numpy.exp(-4*Multijets_emevents["Zeppenfeld_DeltaEta"]**2)

        Multijets_emevents["Rpt"] = Rpt(Muon_collections_2jet, Electron_collections_2jet, [Jet_collections[emevents.nJet30 >= 2][:,0], Jet_collections[emevents.nJet30 >= 2][:,1]])

        Multijets_emevents["pt_cen"] = pt_cen(Muon_collections_2jet, Electron_collections_2jet, [Jet_collections[emevents.nJet30 >= 2][:,0], Jet_collections[emevents.nJet30 >= 2][:,1]])
        Multijets_emevents["pt_cen_Deltapt"] = Multijets_emevents["pt_cen"]/(Jet_collections[emevents.nJet30 >= 2][:,0] - Jet_collections[emevents.nJet30 >= 2][:,1]).pt
        Multijets_emevents["abspt_cen_Deltapt"] = abs(Multijets_emevents["pt_cen_Deltapt"])

        Multijets_emevents["Ht_had"] = ak.sum(Jet_collections[emevents.nJet30 >= 2].pt, 1) + Muon_collections_2jet.pt + Electron_collections_2jet.pt
        Multijets_emevents["Ht"] = ak.sum(Jet_collections[emevents.nJet30 >= 2].pt, 1) + Muon_collections_2jet.pt + Electron_collections_2jet.pt
        return emevents, onejets_emevents, Multijets_emevents

    # we will receive a NanoEvents instead of a coffea DataFrame
    def process(self, events):
        out = self.accumulator.identity()
        emevents = self.Vetos(events)
        emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections = self.Corrections(emevents)
        emevents, onejets_emevents, Multijets_emevents = self.interesting(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections)
        emevents = self.SF(emevents)
        for var in self.var_ :
            out[var+'_0jets'].add( processor.column_accumulator( emevents[emevents.nJet30 == 0][var].to_numpy() ) )

        for var in self.var_ :
            out[var+'_1jets'].add( processor.column_accumulator( emevents[emevents.nJet30 >= 1][var].to_numpy() ) )
        for var in self.var_1jet_ :
            out[var+'_1jets'].add( processor.column_accumulator( onejets_emevents[var].to_numpy() ) )

        for var in self.var_ :
            out[var+'_2jets'].add( processor.column_accumulator( emevents[emevents.nJet30 >= 2][var].to_numpy() ) )
        for var in self.var_1jet_ :
            out[var+'_2jets'].add( processor.column_accumulator( onejets_emevents[onejets_emevents.nJet30 >= 2][var].to_numpy() ) )
        for var in self.var_2jet_ :
            out[var+'_2jets'].add( processor.column_accumulator( Multijets_emevents[var].to_numpy() ) )
 
        return out

    def postprocess(self, accumulator):
        return accumulator

if __name__ == '__main__':
  years = ['2017']
  for year in years:
    with open('lumi_'+year+'.json') as f:
      lumiWeight = json.load(f)
    processor_instance = MyDF(lumiWeight)
    outname = os.path.basename(__file__).replace('.py','')
    save(processor_instance, f'processors/{outname}_{year}.coffea')
