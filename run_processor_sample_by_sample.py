from coffea import processor
from coffea.util import load, save
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from pathlib import Path
import glob, os, json, logging, argparse
import find_samples

import uproot
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

logging.basicConfig(filename='_run_processor.log', level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
rootLogger = logging.getLogger()
logging.captureWarnings(True)


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Run coffea processors')
  parser.add_argument('-p', '--parsl', action='store_true', help='Use parsl to distribute')
  parser.add_argument('-c', '--condor', action='store_true', help='Run on condor')
  parser.add_argument('-b', '--baseprocessor', type=str, default=None, help='processor tag')
  parser.add_argument('-y', '--year', type=str, default=None, help='analysis year')
  parser.add_argument('-j', '--workers', type=int, default=100, help='Number of workers to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
  args = parser.parse_args()

  if args.parsl:
    from parsl_config import parsl_condor_config, parsl_local_config
    import parsl
    executor = processor.parsl_executor
    if args.condor:
      htex = parsl_condor_config(workers=args.workers)
    else:
      ncpu = os.cpu_count()
      print ("Number of cores: %i"%ncpu)
      htex = parsl_local_config(workers=ncpu)
    # keep retrying in case the parsl fails to load
    retry_count = 0
    while True:
      try:
          parsl.load(htex)
          break
      except:
          retry_count += 1
          logging.warning(f'Failed to load parsl. retry {retry_count}')
          time.sleep(10)
  else:
    executor = processor.futures_executor  
    ncpu = os.cpu_count()
    print ("Number of cores: %i"%ncpu)
    executor_args['workers'] = ncpu

  with open(f'lumi_{args.year}.json') as f:
      lumiWeight = json.load(f)

  processorpath = f'processors/{args.baseprocessor}_{args.year}.coffea' 
  processor_instance = load(processorpath)
#  pre_executor = processor.futures_executor #parsl_executor#(config=parsl_local_config(os.cpu_count()))
#  pre_args = {"schema": NanoAODSchema, 'savemetrics': True, 'desc': f'Preprocessing {args.baseprocessor} {args.year} ', 'workers': os.cpu_count()} #'config': parsl_local_config(10)}#, 'workers': os.cpu_count()}
  sample_groups = ['signal']#, 'data', 'diboson', 'top', 'dy', 'higgs', 'ewk']
  outputPath = f"./results/{args.year}/{args.baseprocessor}"
  Path(outputPath).mkdir(parents=True, exist_ok=True)

  import time
  t_start = time.perf_counter()

  for group in sample_groups:
    samples = {}
    for samples_shorthand in find_samples.samples_to_run[group]:
  #    if args.parsl:
       samples[samples_shorthand] = glob.glob(f'/hdfs/store/user/kaho/NanoPost_{args.year}_v1p2/{samples_shorthand}*/*/*/*/*root')
  #    else:
      #samples[samples_shorthand] = glob.glob(f'root://cmsxrootd.hep.wisc.edu//store/user/kaho/NanoPost_{args.year}/{samples_shorthand}*/*/*/*/*root')
    rootLogger.info('Will process: '+' '.join(list(samples.keys()))) 
    executor_args = {"schema": NanoAODSchema, 'savemetrics': True, 'desc': f'Processing {args.baseprocessor} {args.year} {group} '}#, 'config': htex}

    result = processor.run_uproot_job(
      samples,
      "Events",
      processor_instance,
      executor, 
      executor_args,
      chunksize=200000  
      #pre_executor,
      #pre_args
    )
    save(result, f"{outputPath}/output_{group}.coffea")
  t_stop = time.perf_counter()
  print("Elapsed time (s):", t_stop-t_start)  
  if args.parsl:
    parsl.dfk().cleanup()
    parsl.clear()
