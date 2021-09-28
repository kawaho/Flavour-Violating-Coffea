import ROOT as r
import sys
for yr in ['2016', '2017', '2018']:
  QCDFile = r.TFile(f"QCD/htt_scalefactors_legacy_{yr}.root")
  w = QCDFile.Get("w")
  if yr=='2018':
    hss = w.obj("hist_em_qcd_osss_closureOS")
    hos = w.obj("hist_em_qcd_extrap_uncert")
  else:
    hss = w.obj("hist_em_qcd_osss_ss_corr")
    hos = w.obj("hist_em_qcd_osss_os_corr")
  fileOut = r.TFile(f'em_qcd_osss_{yr}.root', 'recreate')
  hss.Write()
  hos.Write()
  fileOut.Close()
  print(f"----------------------Year {yr}----------------------")
  if yr=='2018':
    w.function("em_qcd_osss_binned").dumpFormula()
  else:
    w.function("em_qcd_osss_0jet").dumpFormula()
    w.function("em_qcd_osss_1jet").dumpFormula()
    w.function("em_qcd_osss_2jet").dumpFormula()
