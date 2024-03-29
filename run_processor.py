from coffea import processor
from coffea.util import load, save
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema, BaseSchema
from pathlib import Path
import glob, os, json, logging, argparse
import find_samples
import importlib
import uproot
import sys
sys.path.insert(1, '/afs/crc.nd.edu/user/k/kho2/Flavour-Violating-Coffea/processors')
uproot.open.defaults["xrootd_handler"] = uproot.source.xrootd.MultithreadedXRootDSource

#logging.basicConfig(filename='_run_processor.log', level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#logging.basicConfig(level=logging.INFO, format='%(asctime)s.%(msecs)03d %(levelname)s %(name)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
#rootLogger = logging.getLogger()
#logging.captureWarnings(True)

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Run coffea processors')
  parser.add_argument('-w', '--wq', action='store_true', help='Use work-queue to distribute')
  parser.add_argument('-g', '--group', type=str, default=None, help='Run a group only')
  parser.add_argument('-t', '--test', action='store_true', help='Run on GG_LFV_125 for testing')
  parser.add_argument('-b', '--baseprocessor', type=str, default=None, help='processor tag')
  parser.add_argument('-r', '--remote', action='store_true', help='reading from remote sites')
  parser.add_argument('-s', '--subfix', type=str, default=None, help='output tag')
  parser.add_argument('-o', '--onejob', type=str, default=None, help='specify one job to run')
  parser.add_argument('-bs', '--processorsubfix', type=str, default=None, help='processor tag')
  parser.add_argument('-y', '--year', type=str, default='2017', help='analysis year')
  parser.add_argument('-j', '--workers', type=int, default=25, help='Number of workers to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
  parser.add_argument('-c', '--nchunks', type=int, default=None, help='Max number of chunks to run')
  args = parser.parse_args()

  executor_args = {"schema": NanoAODSchema, 'savemetrics': True, 'desc': f'Processing {args.baseprocessor} {args.year} '}#, 'config': htex}

  substring = ""
  if args.subfix:
    substring+=f"_{args.subfix}"
  if args.onejob:
    substring+=f"_{args.onejob}"
  if args.group:
    substring+=f"_{args.group}"

  if args.wq:
    master_name = '{}-wq-coffea-'.format(os.environ['USER'])+args.baseprocessor+'-'+args.year+substring 
    executor = processor.work_queue_executor
    #print(f"Please submit: condor_submit_workers --cores 8 --memory 8000 --disk 4000 -M "+'{}-wq-coffea-'.format(os.environ['USER'])+args.baseprocessor+'-'+args.year+" 25")
    executor_args = {
    "schema": NanoAODSchema, 
    # give the manager a name so workers can automatically find it:
    'master_name': master_name,
    # find a port to run work queue in this range (above 1024):
    'port': [9123,9130],
    # if not given, assume environment already setup at execution site
    'environment_file': "remote-coffea-env.tar.gz",
    # adjust size of resources allocated to tasks as they are measured
    # use maximum resources seen, retry on maximum values if exhausted.
    'resource_monitor': True,
    'resources_mode': 'auto',
    'extra_input_files': ['processors/kinematics.py', 'processors/BDT_functions.py', 'processors/Corrections.py', 'processors/Vetos.py'],
    #'extra_input_files': ['em_qcd_osss_2016.root', 'em_qcd_osss_2017.root', 'em_qcd_osss_2018.root'],
  #  # print messages when tasks are submitted, and as they return, their
  #  # resource allocation and usage.
    #'verbose': True,
    # detailed debug messages
    #'debug_log': 'debug.log',
    # lifetime of tasks, workers, and resource allocations
    #'transactions_log': 'tr.log',
    # time series of manager statistics, plot with work_queue_graph_log
    #'stats_log': 'stats.log',
    }
#    # no task can use more than these maximum values:
#    'cores': 1,
#    'memory': 8000,
#    'disk': 8000,
    os.system("condor_submit_workers --cores 8 --memory 8000 --disk 8000 -M "+master_name+f" {args.workers}")
  else:
    executor = processor.futures_executor  
    ncpu = int(os.cpu_count())
    print ("Number of cores: %i"%ncpu)
    print ("Using %i"%args.workers)
    executor_args['workers'] = args.workers

  with open(f'lumi_{args.year}.json') as f:
      lumiWeight = json.load(f)

  samples = {}

  if args.test:
    samples_to_run = ['GluGlu_LFV_HToEMu_M125']
  elif args.onejob:
    samples_to_run = [args.onejob]
  elif args.group:
    samples_to_run = find_samples.samples_to_run[args.group]
  else:
    samples_to_run = find_samples.samples_to_run[args.baseprocessor]

  if args.remote:
    #read from wisc
    with open(f'samples_{args.year}.json') as f:
        all_samples = json.load(f)
    for samples_shorthand in samples_to_run:
    #for samples_shorthand in find_samples.samples_to_run[args.baseprocessor]:
        samples[samples_shorthand] = all_samples[samples_shorthand]
  else:
    for samples_shorthand in lumiWeight:
      if samples_shorthand in samples_to_run:
        samples[samples_shorthand] = [i.replace("/hadoop", "root://deepthought.crc.nd.edu/") for i in glob.glob(f'/hadoop/store/user/kaho/NanoPost_{args.year}_v1p3/{samples_shorthand}*/*/*/*/*root')]
        #samples[samples_shorthand] = glob.glob(f'/hadoop/store/user/kaho/NanoPost_{args.year}_v1p3/{samples_shorthand}*/*/*/*/*root')

    if 'data' in samples_to_run:
      samples['data'] = [i.replace("/hadoop", "root://deepthought.crc.nd.edu/") for i in glob.glob(f'/hadoop/store/user/kaho/NanoPost_{args.year}_v1p3/SingleMuon/*/*/*/*root')]

#  rootLogger.info('Will process: '+' '.join(list(samples.keys()))) 
  processorpath = f'processors/{args.baseprocessor}_{args.year}.coffea' 
  processor_instance = load(processorpath)
  if 'DF' in args.baseprocessor:
    processor_instance.sample_list(*find_samples.samples_to_run[args.baseprocessor])
  result = processor.run_uproot_job(
      samples,
      "Events",
      processor_instance,
      executor, 
      executor_args,
      maxchunks=args.nchunks,
      chunksize=80000,
      #dynamic_chunksize=True
      #dynamic_chunksize=True
      #pre_executor,
      #pre_args
  )
  outputPath = f"./results/{args.year}/{args.baseprocessor}"
  Path(outputPath).mkdir(parents=True, exist_ok=True)
  save(result, f"{outputPath}/output"+substring+".coffea")
