from coffea import processor
from coffea.util import load, save
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from pathlib import Path
import glob, os, json, logging, argparse
import find_samples
import importlib
import uproot
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

#logging.basicConfig(filename='_run_processor.log', level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#rootLogger = logging.getLogger()
#logging.captureWarnings(True)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Run coffea processors')
  parser.add_argument('-w', '--wq', action='store_true', help='Use work-queue to distribute')
  parser.add_argument('-t', '--test', action='store_true', help='Run on GG_LFV_125 for testing')
  parser.add_argument('-b', '--baseprocessor', type=str, default=None, help='processor tag')
  parser.add_argument('-s', '--subfix', type=str, default=None, help='output tag')
  parser.add_argument('-y', '--year', type=str, default='2017', help='analysis year')
  parser.add_argument('-j', '--workers', type=int, default=200, help='Number of workers to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
  parser.add_argument('-c', '--nchunks', default=None, help='Max number of chunks to run')
  args = parser.parse_args()
  nchunks = int(args.nchunks) if not args.nchunks is None else args.nchunks
  executor_args = {"schema": NanoAODSchema, 'savemetrics': True, 'desc': f'Processing {args.baseprocessor} {args.year} '}#, 'config': htex}

  if args.wq:
    executor = processor.work_queue_executor
    executor_args = {
    "schema": NanoAODSchema, 
    # give the manager a name so workers can automatically find it:
    'master_name': '{}-wq-coffea'.format(os.environ['USER']),
    # find a port to run work queue in this range (above 1024):
    'port': [9123,9130],
    # if not given, assume environment already setup at execution site
    'environment_file': "remote-coffea-env.tar.gz",
    # adjust size of resources allocated to tasks as they are measured
    # use maximum resources seen, retry on maximum values if exhausted.
    'resource_monitor': True,
    'resources_mode': 'auto',
  #  # print messages when tasks are submitted, and as they return, their
  #  # resource allocation and usage.
  #  'verbose': True,
  #  # detailed debug messages
  #  'debug_log': 'debug.log',
  #  # lifetime of tasks, workers, and resource allocations
  #  'transactions_log': 'tr.log',
  #  # time series of manager statistics, plot with work_queue_graph_log
  #  'stats_log': 'stats.log',
    }
#    # no task can use more than these maximum values:
#    'cores': 1,
#    'memory': 8000,
#    'disk': 8000,

  else:
    executor = processor.futures_executor  
    ncpu = int(os.cpu_count()/2)
    print ("Number of cores: %i"%ncpu)
    executor_args['workers'] = ncpu

  with open(f'lumi_{args.year}.json') as f:
      lumiWeight = json.load(f)

  samples = {}

  if args.test:
    samples_to_run = ['GluGlu_LFV_HToEMu_M125']
  else:
    samples_to_run = find_samples.samples_to_run[args.baseprocessor]

  for samples_shorthand in lumiWeight:
    if samples_shorthand in samples_to_run:
      samples[samples_shorthand] = glob.glob(f'/hadoop/store/user/kaho/NanoPost_{args.year}_v1p3/{samples_shorthand}*/*/*/*/*root')

  if 'data' in samples_to_run:
    samples['data'] = glob.glob(f'/hadoop/store/user/kaho/NanoPost_{args.year}_v1p3/SingleMuon/*/*/*/*root')

  #tmp solution to read from wisc
  #with open(f'data.json') as f:
  #    all_samples = json.load(f)
  #for samples_shorthand in lumiWeight:
  #  if samples_shorthand in find_samples.samples_to_run[args.baseprocessor]:
  #    samples[samples_shorthand] = all_samples[samples_shorthand]

#  rootLogger.info('Will process: '+' '.join(list(samples.keys()))) 
  processorpath = f'processors/{args.baseprocessor}_{args.year}.coffea' 
  processor_instance = load(processorpath)
  processor_instance.sample_list(*find_samples.samples_to_run[args.baseprocessor])
  result = processor.run_uproot_job(
      samples,
      "Events",
      processor_instance,
      executor, 
      executor_args,
      chunksize=20000,
      maxchunks=nchunks 
      #pre_executor,
      #pre_args
  )
  outputPath = f"./results/{args.year}/{args.baseprocessor}"
  Path(outputPath).mkdir(parents=True, exist_ok=True)
  if args.subfix:
    save(result, f"{outputPath}/output_{args.subfix}.coffea")
  else:
    save(result, f"{outputPath}/output.coffea")
