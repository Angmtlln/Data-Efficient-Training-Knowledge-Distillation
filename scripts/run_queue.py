"""Sequential CLI for prepared selections. No remote host or credentials required."""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--part',choices=['01_cpu_selection','02_hypotheses','03_followups'],required=True)
    p.add_argument('--methods',nargs='+');p.add_argument('--seeds',nargs='+',type=int,default=None)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--local',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('--pilot',action='store_true')
    a=p.parse_args();a.seeds=a.seeds or ([202,303,404] if a.part=='03_followups' else [17,42,101]);project=ROOT/'experiments'/a.part
    groups=json.loads((project/'experiment_groups.json').read_text())
    methods=a.methods or list(dict.fromkeys(m for group in groups.values() for m in group))
    tasks=[]
    for seed in a.seeds:
        for method in methods:
            selection=project/'selections'/f'{method}__s{seed}.json'
            if not selection.is_file():p.error(f'Missing selection: {selection}')
            tasks.append((method,seed,[sys.executable,str(project/'code/train.py'),'--selection',str(selection),
                '--seed',str(seed),'--persistent',str(a.output.resolve()),'--local',str(a.local.resolve())]+(['--pilot'] if a.pilot else [])))
    if a.dry_run:
        print(json.dumps({'count':len(tasks),'tasks':[(m,s) for m,s,_ in tasks]},indent=2));return
    a.output.mkdir(parents=True,exist_ok=True);a.local.mkdir(parents=True,exist_ok=True)
    # The same part cannot run two queues at once, even with different output paths.
    lock=(project/'.training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.environ.update(USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false')
    logs=a.output/'logs';logs.mkdir(exist_ok=True)
    for method,seed,cmd in tasks:
        print(f'START {method} seed={seed}',flush=True)
        with (logs/f'{method}_s{seed}.log').open('a') as stream:
            result=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise SystemExit(f'Failed: {method} seed={seed}. Inspect {logs}; repeat command to resume.')
        print(f'COMPLETE {method} seed={seed}',flush=True)
        subprocess.run([sys.executable,str(project/'code/report.py'),'--output',str(a.output)],check=True)
if __name__=='__main__':main()
