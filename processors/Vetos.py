import awkward as ak
def Vetos(_year, events, sameCharge=False):
    if _year == '2016preVFP':
      mpt_threshold = 26
      trigger = events.HLT.IsoMu24 | events.HLT.IsoTkMu24
    elif _year == '2016postVFP':
      mpt_threshold = 26
      trigger = events.HLT.IsoMu24 | events.HLT.IsoTkMu24
    elif _year == '2017':
      mpt_threshold = 29
      trigger = events.HLT.IsoMu27
    elif _year == '2018':
      mpt_threshold = 26
      trigger = events.HLT.IsoMu24

    #Choose em channel and IsoMu Trigger
    emevents = events[(events.channel == 0) & (trigger == 1)]

    E_collections = emevents.Electron
    M_collections = emevents.Muon

    #Kinematics Selections
    emevents["Electron", "Target"] = ((E_collections.pt > 20) & (abs(E_collections.eta) < 2.5) & (abs(E_collections.dxy) < 0.05) & (abs(E_collections.dz) < 0.2) & (E_collections.convVeto) & (E_collections.mvaFall17V2noIso_WP80) & (E_collections.pfRelIso03_all < 0.1) & (E_collections.lostHits<2))
    emevents["Muon", "Target"] = ((M_collections.pt > mpt_threshold) & (abs(M_collections.eta) < 2.4) & (abs(M_collections.dxy) < 0.05) & (abs(M_collections.dz) < 0.2) & (M_collections.tightId) & (M_collections.pfRelIso04_all < 0.15))

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
