import os
import getpass

from parsl.providers import LocalProvider, CondorProvider
from parsl.channels import LocalChannel
from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.addresses import address_by_hostname

UID = os.getuid()
UNAME = getpass.getuser()

def parsl_local_config(workers=1):
    log_dir = 'parsl_logs'

    htex = Config(
        executors=[
            HighThroughputExecutor(
                label="coffea_parsl_default",
                cores_per_worker=1,
                max_workers=workers,
                worker_logdir_root=log_dir,
                provider=LocalProvider(
                    channel=LocalChannel(),
                    init_blocks=1,
                    max_blocks=1,
                ),
            )
        ],
        strategy=None,
        run_dir=os.path.join(log_dir,'runinfo'),
        retries = 2,
    )
    return htex

def parsl_condor_config(workers=1):

    x509_proxy = f'x509up_u{UID}'
    grid_proxy_dir = '/tmp'

    cores_per_job = 1
    mem_per_core = 2000
    mem_request = mem_per_core * cores_per_job
    init_blocks = 1
    min_blocks = 1
    max_blocks = workers
    htex_label='coffea_parsl_condor_htex'
    log_dir = 'parsl_logs'
    log_dir_full = os.path.join('/nfs_scratch/'+os.environ['LOGNAME'],log_dir)

    worker_init = f'''
echo "Setting up environment"
tar -zxf coffeaenv.tar.gz
source coffeaenv/bin/activate
export PATH=coffeaenv/bin:$PATH
export PYTHONPATH=coffeaenv/lib/python3.7/site-packages:$PYTHONPATH
export X509_USER_PROXY={x509_proxy}
mkdir -p {log_dir}/{htex_label}
echo "Environment ready"
'''

    # requirements for T2_US_Wisconsin (HAS_CMS_HDFS forces to run a T2 node not CHTC)
    scheduler_options = f'''
transfer_output_files   = {log_dir}/{htex_label}
RequestMemory           = {mem_request}
RequestCpus             = {cores_per_job}
+RequiresCVMFS          = True
Requirements            = TARGET.HAS_CMS_HDFS && TARGET.Arch == "X86_64"
notification            = Complete
'''
#priority                = 10
#TARGET.HAS_CMS_HDFS
    transfer_input_files = [os.path.join(os.path.dirname(os.path.abspath(__file__)),'coffeaenv.tar.gz'), os.path.join(grid_proxy_dir, x509_proxy)]

    htex = Config(
        executors=[
            HighThroughputExecutor(
                label=htex_label,
                address=address_by_hostname(),
                prefetch_capacity=0,
                cores_per_worker=1,
                max_workers=cores_per_job,
                worker_logdir_root=log_dir,
                provider=CondorProvider(
                    channel=LocalChannel(
                        userhome='/nfs_scratch/'+os.environ['LOGNAME'],
                    ),
                    init_blocks=init_blocks,
                    min_blocks=min_blocks,
                    max_blocks=max_blocks,
                    nodes_per_block=1,
                    worker_init=worker_init,
                    transfer_input_files=transfer_input_files,
                    scheduler_options=scheduler_options,
                ),
            ),
            HighThroughputExecutor(
                label="coffea_parsl_default",
                cores_per_worker=1,
                max_workers=os.cpu_count(), #multicore local
                worker_logdir_root=log_dir,
                provider=LocalProvider(
                    channel=LocalChannel(),
                    init_blocks=1,
                    max_blocks=1,
                ),
            ),
        ],
        strategy='simple',
        run_dir=os.path.join(log_dir_full,'runinfo'),
        retries = 2, # retry all failures, xrootd failures are retried then skipped via coffea executor itself
    )

    return htex
