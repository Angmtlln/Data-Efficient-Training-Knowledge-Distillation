"""Sequential, resumable queue. Never stops other GPU processes."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from common import dump

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--local',type=Path,required=True)
    p.add_argument('--seeds',nargs='+',type=int,default=[1103,1201,1301]);p.add_argument('--methods',nargs='+',choices=['roots','mixed_independent','mixed_paired','mixed_rank'],default=['roots','mixed_independent','mixed_paired','mixed_rank'])
    p.add_argument('--dry-run',action='store_true');p.add_argument('--pilot-only',action='store_true');p.add_argument('--wait-seconds',type=int,default=30);a=p.parse_args()
    tasks=[(m,s) for s in a.seeds for m in a.methods]
    if a.dry_run:print(json.dumps(tasks));return
    subprocess.run([sys.executable,str(ROOT/'verify.py')],check=True)
    a.output=a.output.resolve();a.local=a.local.resolve();a.output.mkdir(parents=True,exist_ok=True);a.local.mkdir(parents=True,exist_ok=True)
    lock=(a.output/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
    def run(method,seed,pilot):
        while True:
            active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if not active:break
            dump(a.output/'state.json',dict(status='waiting_for_gpu',active_pids=active.splitlines()));time.sleep(a.wait_seconds)
        if shutil.disk_usage(a.local).free<10*1024**3:raise RuntimeError('Need 10 GiB free for checkpoints and models')
        dump(a.output/'state.json',dict(status='training',method=method,seed=seed,pilot=pilot))
        cmd=[sys.executable,str(ROOT/'code/train.py'),'--selection',str(ROOT/'selections'/f'{method}__s{seed}.json'),'--seed',str(seed),'--persistent',str(a.output),'--local',str(a.local)]
        if pilot:cmd.append('--pilot')
        logs=a.output/'logs';logs.mkdir(exist_ok=True)
        with (logs/f'{method}_{seed}_{"pilot" if pilot else "full"}.log').open('a') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        # Exercise both independent and paired loss paths before full training.
        for method in ['roots','mixed_rank']:run(method,a.seeds[0],True)
        if not a.pilot_only:
            for method,seed in tasks:
                run(method,seed,False)
                subprocess.run([sys.executable,str(ROOT/'analyze.py'),'--output',str(a.output)],check=True)
        dump(a.output/'state.json',dict(status='complete',pilot_only=a.pilot_only))
    except Exception as e:dump(a.output/'state.json',dict(status='failed',error=str(e)));raise
if __name__=='__main__':main()
