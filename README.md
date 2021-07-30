# Flavour-Violating-Coffea
This is a repository to run LFV H -> e + mu analysis with Coffea
# Setup
Create a cvmfs-based portable virtual env and install Coffea
```bash
source setup.sh
```
If you experienced problems with cvmfs (e.g. plot1d() is intolorantly slow), use the cvmfs independent script instead
```bash
source setup_noCVMFS.sh
```
After setting up the first time, just do 
```bash
source coffeaenv/bin/activate (for cvmfs-based portable virtual env)
```
to reactivate the env each time

# Lumi Weights for MC samples
To create json files that contain the MC lumi weights, do
```bash
python find_lumis.py
```

