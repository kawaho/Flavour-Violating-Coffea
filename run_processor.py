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
  args = parser.parse_args()

  executor = processor.futures_executor

  if args.parsl:
    executor = processor.parsl_executor
    if args.condor:
      htex = parsl_condor_config(workers=100)
    else:
      htex = parsl_local_config(workers=100)
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
  
  with open(f'lumi_{args.year}.json') as f:
      lumiWeight = json.load(f)

  ncpu = os.cpu_count()
  print ("Number of cores: %i"%ncpu)

  samples = {}
  
  for samples_shorthand in lumiWeight:
    if samples_shorthand in find_samples.samples_to_run[args.baseprocessor]:
      samples[samples_shorthand] = glob.glob('/hdfs/store/user/kaho/NanoPost1/'+samples_shorthand+'*/*/*/*/*root')

  if 'data' in find_samples.samples_to_run:
    samples['data'] = glob.glob('/hdfs/store/user/kaho/NanoPost1/SingleMuon/*/*/*/*root')

  rootLogger.info('Will process: '+' '.join(list(samples.keys()))) 

  processorpath = f'processors/{args.baseprocessor}_{args.year}.coffea' 
  processor_instance = load(processorpath)
  result = processor.run_uproot_job(
      samples,
      "Events",
      processor_instance,
      executor, 
      {"schema": NanoAODSchema, 'savemetrics': True, 'desc': f'Processing {args.baseprocessor} {args.year} '} #, "workers": os.cpu_count()}
  )
  outputPath = f"/afs/hep.wisc.edu/home/kaho/NDHiggs/results/{args.year}/{args.baseprocessor}"
  Path(outputPath).mkdir(parents=True, exist_ok=True)
  save(result, f"{outputPath}/output.coffea")

  if args.parsl:
    parsl.dfk().cleanup()
    parsl.clear()
