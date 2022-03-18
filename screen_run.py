import argparse
parser = argparse.ArgumentParser(description='Run coffea processors with screen')
parser.add_argument('-s', '--subfix', type=str, default=None, help='output tag')
parser.add_argument('-y', '--year', type=str, default='2016preVFP,2016postVFP,2017,2018', help='analysis year separated by comma')
parser.add_argument('-b', '--baseprocessor', type=str, default=None, help='processor tag separated by comma')
args = parser.parse_args()
f = open("job.sh", "w")
for bp in args.baseprocessor.split(","):
  for yr in args.year.split(","):
    if args.subfix:
      f.write(f'python run_processor.py -b {bp} -y {yr} -r -w -s {args.subfix} &\n')
      print(f'python run_processor.py -b {bp} -y {yr} -r -w -s {args.subfix} &\n')
    else:
      f.write(f'python run_processor.py -b {bp} -y {yr} -r -w &\n')
      print(f'python run_processor.py -b {bp} -y {yr} -r -w &\n')
    f.write('sleep 3\n')
f.close()

