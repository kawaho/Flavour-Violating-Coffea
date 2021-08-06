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
to reactivate the env each time. Use `deactivate` to deactivate (duhhhh!)

## Setup with Conda (for python below 3.7)
To run the plotting script, we would need matplotlib>=3.4.2 which requires python3.7 or above. Easiest way to install python>=3.7 on a shared linux machine is through conda
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
```
You might want to login again or do `source ~/.bashrc`. Then create a conda env and install coffea
```bash
conda create --name coffeaenv_conda python
conda activate coffeaenv_conda
conda install -y -c conda-forge conda-pack xrootd coffea
```
After setting up the first time, just do 
```bash
conda activate coffeaenv_conda
```
to reactivate the env each time. Use `conda deactivate` to deactivate (duhhhh x2!)

# Lumi Weights for MC samples
To create json files that contain the MC lumi weights, do
```bash
python find_lumis.py
```

# Workflow
Workflow of the package is inspired by and configuarations of parsl are modified from https://github.com/dntaylor/NanoAnalysis. All credits to Devin Taylor.

## Coffea Processors
Create coffea processors in the `./processors` directory. For example, do 
```bash
python processors/make1dHist.py 
```
to generate `./processors/make1dHist_{year}.coffea`

## Sample Selections
Modify the dictionary in `find_samples.py` to tell the processors which samples you want to run over. For example, if I want to run over `GluGlu_LFV_HToEMu` with a processor named `makeHist`
```python
samples_to_run = {'makeHist': ['GluGlu_LFV_HToEMu']}
```

## Running Processors
See options in `run_processor.py` to specify the processor to run, the executor (parsl or local only futures_executor) to be used as well as whether or not to run it with condor. Results will be stored in `./results/{year}/{processor name}/*.coffea`

# Format Convesions
It is often useful to convert outputs of coffea to more familiar formats, e.g. csv or ROOT files
1. `dict2csv.py` converts "tuple" like output to csv files
2. `dict2ttree.py` converts "tuple" like output to TTree 
3. `hist2TH.py` converts coffea.hist to THist (use with cvmfs independent env only) 



