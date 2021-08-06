from coffea import processor
from coffea.util import load, save
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from parsl_config import parsl_condor_config, parsl_local_config
from pathlib import Path
import glob, os, json, parsl, logging, argparse
import find_samples

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

  executor_args = {"schema": NanoAODSchema, 'savemetrics': True, 'desc': f'Processing {args.baseprocessor} {args.year} '}

  if args.parsl:
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

  samples = {}
  
  for samples_shorthand in lumiWeight:
    if samples_shorthand in find_samples.samples_to_run[args.baseprocessor]:
      samples[samples_shorthand] = glob.glob(f'/hdfs/store/user/kaho/NanoPost_{args.year}/{samples_shorthand}*/*/*/*/*root')

  if 'data' in find_samples.samples_to_run[args.baseprocessor]:
    samples['data'] = glob.glob('/hdfs/store/user/kaho/NanoPost_{args.year}/SingleMuon/*/*/*/*root')

  rootLogger.info('Will process: '+' '.join(list(samples.keys()))) 

  processorpath = f'processors/{args.baseprocessor}_{args.year}.coffea' 
  processor_instance = load(processorpath)
  result = processor.run_uproot_job(
      samples,
      "Events",
      processor_instance,
      executor, 
      executor_args      
  )
  outputPath = f"./results/{args.year}/{args.baseprocessor}"
  Path(outputPath).mkdir(parents=True, exist_ok=True)
  save(result, f"{outputPath}/output.coffea")

  if args.parsl:
    parsl.dfk().cleanup()
    parsl.clear()
