import awkward as ak
def Vetos(_year, events, sameCharge=False):
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

    #Choose em channel and IsoMu+IsoEle Trigger
    emevents = events[(events.channel == 0) & (events.mtrigger|events.etrigger)]
    E_collections = emevents.Electron
    M_collections = emevents.Muon

    #Kinematics Selections
    E_collections.Target = ((abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.05) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
    M_collections.Target = ((abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.05) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

    #Apply pT cut according to trigger types
    emevents["Electron", "Target_m"] = ((E_collections.pt > ept_threshold[0]) & E_collections.Target)
    emevents["Muon", "Target_m"] = ((M_collections.pt > mpt_threshold[0]) & (M_collections.Target))
    emevents["Electron", "Target_e"] = ((E_collections.pt > ept_threshold[1]) & E_collections.Target)
    emevents["Muon", "Target_e"] = ((M_collections.pt > mpt_threshold[1]) & (M_collections.Target))

    #Muon Trig Matching
    M_collections = M_collections[emevents.Muon.Target_m==1]
    M_collections = ak.pad_none(M_collections, 1, axis=-1)
    #https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L60
    trg_collections = emevents.TrigObj[emevents.TrigObj.id == 13]
    mtrg_Match = (ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1) & emevents.mtrigger)
    mtrg_Match = ak.fill_none(mtrg_Match, False)

    #Electron Trig Matching
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
    #Trig Matching
    M_collections = emevents.Muon
    trg_collections = emevents.TrigObj

    M_collections = M_collections[M_collections.Target==1]
    #https://github.com/cms-sw/cmssw/blob/master/PhysicsTools/NanoAOD/python/triggerObjects_cff.py#L60
    trg_collections = trg_collections[trg_collections.id == 13]

    trg_Match = ak.any((M_collections[:,0].delta_r(trg_collections) < 0.5),1)

    return emevents[trg_Match]
