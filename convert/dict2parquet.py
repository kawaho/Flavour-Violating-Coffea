from coffea.util import load
import numpy as np
import pandas as pd
import glob, os, json, argparse
parser = argparse.ArgumentParser(description='Convert coffea output to csv files')
parser.add_argument('-b', '--baseprocessor', type=str, default='makeDF', help='processor tag (default: %(default)s)')
parser.add_argument('-s', '--subfix', type=str, default=None, help='subfix for the processor tag (default: %(default)s)')
parser.add_argument('-g', '--group', type=str, default='', help='group of subsamples, separated by comma (default: %(default)s)')
#parser.add_argument('-g', '--group', type=str, default='diboson,others,tt,signal,signalAlt,data', help='group of subsamples, separated by comma (default: %(default)s)')
parser.add_argument('-o', '--output', type=str, default='parquet', help='output type (default: %(default)s)')
parser.add_argument('-y', '--year', type=str, default='2016preVFP,2016postVFP,2017,2018', help='analysis year')
parser.add_argument('-sc', '--samecharge', action='store_true', help='save same charge')
args = parser.parse_args()
years = args.year.split(',')

var_dict = {}
for year in years:
  print(f'Processing {year}')
  for subsample in args.group.split(','):
    inputfileName = f"results/{year}/{args.baseprocessor}/output"
    if args.subfix: inputfileName += f'_{args.subfix}'
    if subsample!='': inputfileName += f'_{subsample}'
    result = load(inputfileName+".coffea")
    if isinstance(result,tuple):
        result = result[0]
    for varName in result:
      if varName in var_dict:
          var_dict[varName] = np.append(var_dict[varName],result[varName].value)
      else:
          var_dict[varName] = result[varName].value

outputfile = args.baseprocessor
if args.subfix: outputfile += f'_{args.subfix}'
for subsample in args.group.split(','):
  if subsample!='': outputfile += f'_{subsample}'
for i in var_dict:
  print(i, len(var_dict[i]))

df = pd.DataFrame(var_dict)
df_oc, df_sc = df[df['opp_charge']==1], df[df['opp_charge']==0]
df_oc_gg, df_oc_vbf = df_oc[df_oc['isVBFcat']==0], df_oc[df_oc['isVBFcat']==1]
df_sc_gg, df_sc_vbf = df_sc[df_sc['isVBFcat']==0], df_sc[df_sc['isVBFcat']==1]
df_to_save = {'oc_gg': df_oc_gg, 'oc_vbf':df_oc_vbf}

if args.samecharge:
  df_to_save['sc_gg'] = df_sc_gg
  df_to_save['sc_vbf'] = df_sc_vbf

for subdf in df_to_save:
  if args.output == 'csv':
    df_to_save[subdf].to_csv(f'results/csv4BDT/{outputfile}_{subdf}.csv', index=False)    
  elif args.output == 'parquet':
    df_to_save[subdf].to_parquet(f'results/csv4BDT/{outputfile}_{subdf}.parquet', index=False)    
  print(f'results/csv4BDT/{outputfile}_{subdf}.parquet is created!')
