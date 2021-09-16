# Flavour-Violating-Coffea
This is a repository to run LFV H -> e + mu analysis with Coffea on Notre Dame CRC servers
# Setup
Download miniconda and all the needed modules
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh > conda-install.sh
bash conda-install.sh
unset PYTHONPATH
conda create --name my-coffea-env
conda activate my-coffea-env
conda install -c conda-forge coffea xrootd ndcctools dill conda-pack xgboost scikit-learn root 
conda install -c anaconda ipykernel
```
To minimize size of the tarball to be sent to the remote workers for batch jobs, create another env named  ```remote-coffea-env``` and skip ```scikit-learn root``` as well as ```ipykernel```. After setting up the first time, just do 
```bash
source activate.sh
```
to reactivate the env each time. Use `conda deactivate` to deactivate.

# Lumi json for datasets
To create lumi json files for datasets (input for brilcalc: https://twiki.cern.ch/twiki/bin/view/CMS/BrilcalcQuickStart), do
```bash
python make_lumiMask.py
```

# Lumi Weights for MC samples
To create json files that contain the MC lumi weights, do
```bash
python find_lumis.py
```

# Workflow

## Coffea Processors
Create coffea processors in the `processors` directory. 

## Sample Selections
Modify the dictionary in `find_samples.py` to tell the processors which samples you want to run over. For example, if I want to run over `GluGlu_LFV_HToEMu` with a processor named `makeHist`
```python
samples_to_run = {'makeHist': ['GluGlu_LFV_HToEMu']}
```

## Running Processors
See options in `run_processor.py` to specify the processor to run, the executor (WQ or local only futures_executor) to be used as well as whether or not to run it with condor. Results will be stored in `./results/{year}/{processor name}/*.coffea`

# Format Convesions
It is often useful to convert outputs of coffea to more familiar formats, e.g. csv or ROOT files
1. `dict2parquet.py` converts "tuple" like output to csv or parquet files
2. `dict2ttree.py` converts "tuple" like output to TTree 
3. `hist2TH.py` converts coffea.hist to THist (use with cvmfs independent env only) 

# Experiments
To play around and experiment, use the juypter notebooks in the notebooks directory



