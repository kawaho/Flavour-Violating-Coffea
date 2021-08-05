# Flavour-Violating-Coffea
This is a repository to run LFV H -> e + mu analysis with Coffea on T2_US_Wisconsin servers
# Setup
Create a cvmfs-based portable virtual env and install Coffea
```bash
source setup.sh
```
If you experienced problems with cvmfs (e.g. plot1d() is intolorantly slow) and do not intend to run jobs on condor, use the cvmfs independent script instead
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

# Workflow
Workflow of the package is inspired by and configuarations of parsl are taken from https://github.com/dntaylor/NanoAnalysis. All credits to Devin Taylor.

## Coffea Processors
Create coffea processors in the `./processors` directory. For example, do 
```bash
python processors/make1dHist.py 
```
to generate `./processors/make1dHist_{year}.coffea`

## Sample Selections
Modify the dictionary in `find_samples.py` to tell the processors which samples you want to run over. For example, if I want to run over `GluGlu_LFV_HToEMu` with a processor named `make1dHist`
```python
samples_to_run = {'make1dHist': ['GluGlu_LFV_HToEMu']}
```

## Running Processors
See options in `run_processor.py` to specify the processor to run, the executor (parsl or local only futures_executor) to be used as well as whether or not to run it with condor. Results will be stored in `./results/{year}/*.coffea`

# Format Convesions
It is often useful to convert outputs of coffea to more familiar formats, e.g. csv or ROOT files
1. `dict2csv.py` converts "tuple" like output to csv files
2. `dict2ttree.py` converts "tuple" like output to TTree 
3. `hist2TH.py` converts coffea.hist to THist (use with cvmfs independent env only) 



