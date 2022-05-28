import awkward as ak
import numpy
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

def Zeppenfeld(lep1, lep2, jets):
    emVar = lep1+lep2
    if len(jets)==1:
        return emVar.eta - (jets[0].eta)/2
    elif len(jets)==2:
        return emVar.eta - (jets[0].eta + jets[1].eta)/2
    else:
        return -999
    
def mT(met, lep1):
    return numpy.sqrt(abs((numpy.sqrt(lep1.mass**2+lep1.pt**2) + met.pt)**2 - (lep1+met).pt**2))

def mT3(met, lep1, lep2):
    lep12 = lep1+lep2
    return numpy.sqrt(abs((numpy.sqrt(lep12.mass**2+lep12.pt**2) + met.pt)**2 - (lep12+met).pt**2))

def pt_cen(lep1, lep2, jets):
    emVar = lep1+lep2
    if len(jets)==1:
        return emVar.pt - jets[0].pt/2
    elif len(jets)==2:
        return emVar.pt - (jets[0] + jets[1]).pt/2
    else:
        return -999

def interestingKin(emevents, Electron_collections, Muon_collections, MET_collections, Jet_collections, doSys=False):
    #make interesting variables
    #zero/any no. of jets
    emVar = Electron_collections + Muon_collections
    emevents["e_m_Mass"] = emVar.mass
    emevents["empt"] = emVar.pt
    emevents["ept"] = Electron_collections.pt
    emevents["mpt"] = Muon_collections.pt
    emevents["eEta"] = Electron_collections.eta
    emevents["mEta"] = Muon_collections.eta
    emevents["DeltaEta_e_m"] = abs(Muon_collections.eta - Electron_collections.eta)
    emevents["met"] = MET_collections.pt
    emevents["DeltaPhi_em_met"] = emVar.delta_phi(MET_collections)
    emevents["njets"] = emevents.nJet30 
#     emevents["e_met_mT"] = mT(Electron_collections, MET_collections)
#     emevents["m_met_mT"] = mT(Muon_collections, MET_collections)
#     pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
#     emevents["pZeta85"] = pZeta_ - 0.85*pZetaVis_

    #1 jet
    emevents['j1pt'] = ak.fill_none(Jet_collections[:,0].pt, 0)
    emevents['j1Eta'] = Jet_collections[:,0].eta
    emevents["DeltaEta_j1_em"] = abs(Jet_collections[:,0].eta - emVar.eta)

    #2 or more jets
    emevents['j2pt'] = ak.fill_none(Jet_collections[:,1].pt, 0)
    emevents['j2Eta'] = Jet_collections[:,1].eta
    emevents["j1_j2_mass"] = ak.fill_none((Jet_collections[:,0] + Jet_collections[:,1]).mass, 0)
    emevents["DeltaEta_em_j1j2"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - emVar.eta)
    emevents["DeltaEta_j1_j2"] = abs(Jet_collections[:,0].eta - Jet_collections[:,1].eta)
    emevents["isVBFcat"] = ((emevents["njets"] >= 2) & (emevents["j1_j2_mass"] > 400) & (emevents["DeltaEta_j1_j2"] > 2.5)) 
    emevents["isVBFcat"] = ak.fill_none(emevents["isVBFcat"], False)
    emevents["Zeppenfeld_DeltaEta"] = Zeppenfeld(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/emevents["DeltaEta_j1_j2"]
    emevents["Rpt"] = Rpt(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
    emevents["pt_cen_Deltapt"] = pt_cen(Muon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/(Jet_collections[:,0] - Jet_collections[:,1]).pt
    emevents["Ht_had"] = ak.sum(Jet_collections.pt, 1)

    #Systematics
    if doSys:
      for UpDown in ['Up', 'Down']:
        #Electron Energy Scale
#        tmpFac = 1.01 if UpDown=='Up' else 0.99
#        tmpElectron_collections = ak.copy(Electron_collections)
#        tmpElectron_collections['pt'] = tmpElectron_collections['pt']*tmpFac
#
#        tmpElectron_collections = ak.zip(
#        {
#          "pt":   Electron_collections.pt,
#          "eta":  Electron_collections.eta,
#          "phi":  Electron_collections.phi,
#          "energy": Electron_collections.energy,
#          "mass": Electron_collections.mass
#        },
#          with_name="PtEtaPhiELorentzVector",
#        )
#        tmpElectron_collections = tmpElectron_collections*tmpElectron_collections[f'dEscale{UpDown}']/tmpElectron_collections.energy
#        #Redo all Electron var
#        tmpemVar = tmpElectron_collections + Muon_collections
#        emevents[f'e_m_Mass_ees_{UpDown}'] = tmpemVar.mass
#        emevents[f"empt_ees_{UpDown}"] = tmpemVar.pt
#        emevents[f"DeltaEta_e_m_ees_{UpDown}"] = abs(Muon_collections.eta - tmpElectron_collections.eta)
#        emevents["DeltaPhi_em_met_ees_{UpDown}"] = tmpemVar.delta_phi(MET_collections)
#        emevents[f"DeltaEta_j1_em_ees_{UpDown}"] = abs(Jet_collections[:,0].eta - tmpemVar.eta)
#        emevents[f"DeltaEta_em_j1j2_ees_{UpDown}"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - tmpemVar.eta)
#        emevents[f"Zeppenfeld_DeltaEta_ees_{UpDown}"] = Zeppenfeld(Muon_collections, tmpElectron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/emevents["DeltaEta_j1_j2"]
#        emevents[f"Rpt_ees_{UpDown}"] = Rpt(Muon_collections, tmpElectron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
#        emevents[f"pt_cen_Deltapt_ees_{UpDown}"] = pt_cen(Muon_collections, tmpElectron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/(Jet_collections[:,0] - Jet_collections[:,1]).pt
  
        #Electron Energy Reso
        tmpElectron_collections = ak.zip(
          {
            "pt":   Electron_collections.pt,
            "eta":  Electron_collections.eta,
            "phi":  Electron_collections.phi,
            "energy": Electron_collections.energy,
            "mass": Electron_collections.mass
          },
          with_name="PtEtaPhiELorentzVector",
        )
        tmpElectron_collections = tmpElectron_collections*(Electron_collections[f'dEsigma{UpDown}']+tmpElectron_collections.energy)/tmpElectron_collections.energy
        #Redo all Electron var
        tmpemVar = tmpElectron_collections + Muon_collections
        emevents[f'e_m_Mass_eer_{UpDown}'] = tmpemVar.mass
        emevents[f"empt_eer_{UpDown}"] = tmpemVar.pt
        emevents[f"DeltaEta_e_m_eer_{UpDown}"] = abs(Muon_collections.eta - tmpElectron_collections.eta)
        emevents["DeltaPhi_em_met_eer_{UpDown}"] = tmpemVar.delta_phi(MET_collections)
        emevents[f"DeltaEta_j1_em_eer_{UpDown}"] = abs(Jet_collections[:,0].eta - tmpemVar.eta)
        emevents[f"DeltaEta_em_j1j2_eer_{UpDown}"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - tmpemVar.eta)
        emevents[f"Zeppenfeld_DeltaEta_eer_{UpDown}"] = Zeppenfeld(Muon_collections, tmpElectron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/emevents["DeltaEta_j1_j2"]
        emevents[f"Rpt_eer_{UpDown}"] = Rpt(Muon_collections, tmpElectron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
        emevents[f"pt_cen_Deltapt_eer_{UpDown}"] = pt_cen(Muon_collections, tmpElectron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/(Jet_collections[:,0] - Jet_collections[:,1]).pt
  
        #Muon Energy Scale + Reso 
        tmpMuon_collections = ak.copy(Muon_collections)
        tmpMuon_collections['pt'] = tmpMuon_collections[f'corrected{UpDown}_pt']
        #Redo all Muon var
        tmpemVar = Electron_collections + tmpMuon_collections
        emevents[f'e_m_Mass_me_{UpDown}'] = tmpemVar.mass
        #emevents[f"mpt_Per_e_m_Mass_me_{UpDown}"] = tmpMuon_collections.pt/emevents[f'e_m_Mass_me_{UpDown}']
        #emevents[f"ept_Per_e_m_Mass_me_{UpDown}"] = Electron_collections.pt/emevents[f'e_m_Mass_me_{UpDown}']
        emevents[f"empt_me_{UpDown}"] = tmpemVar.pt
        emevents[f"DeltaEta_e_m_me_{UpDown}"] = abs(tmpMuon_collections.eta - Electron_collections.eta)
        #emevents[f"m_met_mT_me_{UpDown}"] = mT(tmpMuon_collections, MET_collections)
        #pZeta_, pZetaVis_ = pZeta(tmpMuon_collections, Electron_collections,  MET_collections.px,  MET_collections.py)
        #emevents[f"pZeta85_me_{UpDown}"] = pZeta_ - 0.85*pZetaVis_
        emevents["DeltaPhi_em_met_me_{UpDown}"] = tmpemVar.delta_phi(MET_collections)
        emevents[f"DeltaEta_j1_em_me_{UpDown}"] = abs(Jet_collections[:,0].eta - tmpemVar.eta)
        emevents[f"DeltaEta_em_j1j2_me_{UpDown}"] = abs((Jet_collections[:,0] + Jet_collections[:,1]).eta - tmpemVar.eta)
        emevents[f"Zeppenfeld_DeltaEta_me_{UpDown}"] = Zeppenfeld(tmpMuon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/emevents["DeltaEta_j1_j2"]
        emevents[f"Rpt_me_{UpDown}"] = Rpt(tmpMuon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])
        emevents[f"pt_cen_Deltapt_me_{UpDown}"] = pt_cen(tmpMuon_collections, Electron_collections, [Jet_collections[:,0], Jet_collections[:,1]])/(Jet_collections[:,0] - Jet_collections[:,1]).pt
  
        #Unclustered
        tmpMET_collections = ak.copy(MET_collections)
        tmpMET_collections['phi'] = emevents.MET[f'T1_phi_unclustEn{UpDown}'] 
        tmpMET_collections['pt'] = emevents.MET[f'T1_pt_unclustEn{UpDown}'] 
  
        #Redo all MET var
        emevents[f"met_UnclusteredEn_{UpDown}"] = tmpMET_collections.pt
        emevents["DeltaPhi_em_met_UnclusteredEn_{UpDown}"] = emVar.delta_phi(tmpMET_collections)
        # emevents[f"e_met_mT_UnclusteredEn_{UpDown}"] = mT(Electron_collections, tmpMET_collections)
        # emevents[f"m_met_mT_UnclusteredEn_{UpDown}"] = mT(Muon_collections, tmpMET_collections)
        # pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  tmpMET_collections.px,  tmpMET_collections.py)
        # emevents["pZeta85_UnclusteredEn_{UpDown}"] = pZeta_ - 0.85*pZetaVis_
  
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
             tmpJet_collections = ak.pad_none(tmpJet_collections, 2)
             #MET
             tmpMET_collections = ak.copy(emevents.MET)
             tmpMET_collections['pt'] = emevents.MET[f'T1_pt_{jetUncNoyear}{UpDown}']
             tmpMET_collections['phi'] = emevents.MET[f'T1_phi_{jetUncNoyear}{UpDown}']
             #Redo all Jet/MET var
             emevents[f"met_{jetUnc}_{UpDown}"] = tmpMET_collections.pt
             #emevents[f"e_met_mT_{jetUnc}_{UpDown}"] = mT(Electron_collections, tmpMET_collections)
             #emevents[f"m_met_mT_{jetUnc}_{UpDown}"] = mT(Muon_collections, tmpMET_collections)
             #pZeta_, pZetaVis_ = pZeta(Muon_collections, Electron_collections,  tmpMET_collections.px,  tmpMET_collections.py)
             #emevents[f"pZeta85_{jetUnc}_{UpDown}"] = pZeta_ - 0.85*pZetaVis_
             emevents["DeltaPhi_em_met_{jetUnc}_{UpDown}"] = emVar.delta_phi(tmpMET_collections)
             emevents[f'j1pt_{jetUnc}_{UpDown}'] = ak.fill_none(tmpJet_collections[:,0].pt, 0)
             emevents[f'j1Eta_{jetUnc}_{UpDown}'] = tmpJet_collections[:,0].eta
             emevents[f"DeltaEta_j1_em_{jetUnc}_{UpDown}"] = abs(tmpJet_collections[:,0].eta - emVar.eta)
             emevents[f'j2pt_{jetUnc}_{UpDown}'] = ak.fill_none(tmpJet_collections[:,1].pt, 0)
             emevents[f"j1_j2_mass_{jetUnc}_{UpDown}"] = ak.fill_none((tmpJet_collections[:,0] + tmpJet_collections[:,1]).mass, 0)
             emevents[f"DeltaEta_em_j1j2_{jetUnc}_{UpDown}"] = abs((tmpJet_collections[:,0] + tmpJet_collections[:,1]).eta - emVar.eta)
             emevents[f"DeltaEta_j1_j2_{jetUnc}_{UpDown}"] = abs(tmpJet_collections[:,0].eta - tmpJet_collections[:,1].eta)
             emevents[f"isVBFcat_{jetUnc}_{UpDown}"] = ((emevents[f"njets_{jetUnc}_{UpDown}"] >= 2) & (emevents[f"j1_j2_mass_{jetUnc}_{UpDown}"] > 400) & (emevents[f"DeltaEta_j1_j2_{jetUnc}_{UpDown}"] > 2.5)) 
             emevents[f"isVBFcat_{jetUnc}_{UpDown}"] = ak.fill_none(emevents[f"isVBFcat_{jetUnc}_{UpDown}"], False)
             emevents[f"Zeppenfeld_DeltaEta_{jetUnc}_{UpDown}"] = Zeppenfeld(Muon_collections, Electron_collections, [tmpJet_collections[:,0], tmpJet_collections[:,1]])/emevents[f"DeltaEta_j1_j2_{jetUnc}_{UpDown}"]
             emevents[f"Rpt_{jetUnc}_{UpDown}"] = Rpt(Muon_collections, Electron_collections, [tmpJet_collections[:,0], tmpJet_collections[:,1]])
             emevents[f"pt_cen_Deltapt_{jetUnc}_{UpDown}"] = pt_cen(Muon_collections, Electron_collections, [tmpJet_collections[:,0], tmpJet_collections[:,1]])/(tmpJet_collections[:,0] - tmpJet_collections[:,1]).pt
             emevents[f"Ht_had_{jetUnc}_{UpDown}"] = ak.sum(tmpJet_collections.pt, 1)

    return emevents
