from coffea.util import load
import numpy as np
import pandas as pd
import glob, os, json, argparse
parser = argparse.ArgumentParser(description='Convert coffea output to csv files')
parser.add_argument('-b', '--baseprocessor', type=str, default='makeDF', help='processor tag (default: %(default))')
#parser.add_argument('-y', '--year', type=str, default=None, help='analysis year')
args = parser.parse_args()
years = ['2018','2017']
var_dict = [{}, {}, {}]
for year in years:
  print(f'Processing {year}')
  result = load(f"results/{year}/{args.baseprocessor}/output.coffea")
  if isinstance(result,tuple):
      result = result[0]
  for varName in result:
    for i in range(3):
      if f'{i}jets' in varName:
        if varName.replace(f'_{i}jets','') in var_dict[i]:
          var_dict[i][varName.replace(f'_{i}jets','')] = np.append(var_dict[i][varName.replace(f'_{i}jets','')],result[varName].value)
        else:
          var_dict[i][varName.replace(f'_{i}jets','')] = result[varName].value
for i in range(3):
    df = pd.DataFrame(var_dict[i])
    df.to_csv(f'results/csv4BDT/out_{i}jets.csv', index=False)    

