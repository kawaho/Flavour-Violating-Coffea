#!/bin/bash
[ -f nohup.out ] && rm nohup.out
rm -rf ./parsl_logs/
rm -rf /nfs_scratch/$LOGNAME/coffea_parsl_condor_htex/
rm -rf /nfs_scratch/$LOGNAME/parsl_logs/
rm -f _run_processor.log
