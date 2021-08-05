from coffea.util import load
import pandas as pd
import uproot3
import glob, os, json, argparse
parser = argparse.ArgumentParser(description='Convert coffea output to TTree')
parser.add_argument('-b', '--baseprocessor', type=str, default='makeDF', help='processor tag (default: %(default))')
parser.add_argument('-y', '--year', type=str, default=None, help='analysis year')
args = parser.parse_args()
result = load(f"results/{args.year}/{args.baseprocessor}/output.coffea")
if isinstance(hists,tuple):
    result = result[0]
var_dict = [{}, {}, {}]
var_tree = [{}, {}, {}]
for varName in result:
    if '0jets' in varName:
        var_dict[0][varName.replace('_0jets','')] = result[varName].value
        var_tree[0][varName.replace('_0jets','')] = result[varName].value.dtype
    if '1jets' in varName:
        var_dict[1][varName.replace('_1jets','')] = result[varName].value
        var_tree[1][varName.replace('_1jets','')] = result[varName].value.dtype
    if '2jets' in varName:
        var_dict[2][varName.replace('_2jets','')] = result[varName].value
        var_tree[2][varName.replace('_2jets','')] = result[varName].value.dtype
for i in range(3):
    fout = uproot3.recreate(f'results/{args.year}/{args.baseprocessor}/out_{i}jets_{args.year}.root')
    fout['tree'] = uproot3.newtree(var_tree[i])
    fout['tree'].extend(var_dict[i])
    fout.close()
