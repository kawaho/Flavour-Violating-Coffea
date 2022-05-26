#signal,data,diboson,tt,st,dy,wjets,higgs,ewk
import argparse
import find_samples
parser = argparse.ArgumentParser(description='Run coffea processors with screen')
parser.add_argument('-s', '--subfix', type=str, default=None, help='output tag')
parser.add_argument('-o', '--onejob', action='store_true', help='specify one job to run')
parser.add_argument('-y', '--year', type=str, default='2016preVFP,2016postVFP,2017,2018', help='analysis year separated by comma')
parser.add_argument('-b', '--baseprocessor', type=str, default=None, help='processor tag separated by comma')
parser.add_argument('-j', '--workers', type=int, default=25, help='Number of workers to use for multi-worker executors (e.g. futures or condor) (default: %(default)s)')
parser.add_argument('-g', '--group', type=str, default=None, help='Run a group only, comma separated')
args = parser.parse_args()
f = open("job.sh", "w")
list_of_cmd = []
for bp in args.baseprocessor.split(","):
  for yr in args.year.split(","):
    basecommandstr = f'python run_processor.py -b {bp} -y {yr} -r -w -j {args.workers} '
    if args.subfix:
      basecommandstr += f'-s {args.subfix} '
    if args.onejob:
      for i in find_samples.samples_to_run[args.baseprocessor]:
        commandstr = basecommandstr+f'-o {i} &\n'
        f.write(commandstr)
        f.write('sleep 3\n')
        print(commandstr)
    elif args.group:
      for i in args.group.split(","):
        commandstr = basecommandstr+f'-g {i} &\n'
        f.write(commandstr)
        f.write('sleep 3\n')
        print(commandstr)

    else:
      commandstr = basecommandstr+f' &\n'
      f.write(commandstr)
      f.write('sleep 3\n')
      print(commandstr)
f.close()

