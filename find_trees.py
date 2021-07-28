import os, glob, json
samples_names = glob.glob('/hdfs/store/user/kaho/NanoPost1/*/*')
sample_paths = {}
for name in samples_names:
   tree_paths = glob.glob(name+'/*/*/*root')
   with open("%s.json"%os.path.basename(name), 'w') as f: 
     json.dump(tree_paths, f, indent=2)
