from coffea.util import load
import pandas as pd
import glob, os, json, argparse
parser = argparse.ArgumentParser(description='Convert coffea output to csv files')
parser.add_argument('-b', '--baseprocessor', type=str, default='makeDF', help='processor tag (default: %(default))')
parser.add_argument('-y', '--year', type=str, default=None, help='analysis year')
args = parser.parse_args()
result = load(f"results/{args.year}/{args.baseprocessor}/output.coffea")
if isinstance(result,tuple):
    result = result[0]
var_dict = [{}, {}, {}]
for varName in result:
    if '0jets' in varName:
        var_dict[0][varName.replace('_0jets','')] = result[varName].value
    if '1jets' in varName:
        var_dict[1][varName.replace('_1jets','')] = result[varName].value
    if '2jets' in varName:
        var_dict[2][varName.replace('_2jets','')] = result[varName].value
for i in range(3):
    df = pd.DataFrame(var_dict[i])
    df.to_csv(f'results/{args.year}/{args.baseprocessor}/out_{i}jets_{args.year}.csv')    

