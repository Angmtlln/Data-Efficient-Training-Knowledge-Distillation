"""Validate and summarize completed SST-2 and optional IMDb predictions."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
from datasets import load_from_disk
from sklearn.metrics import accuracy_score,f1_score

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];ROOT=REPO/'results/proxy_mixture'
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'));from common import read_data,digest
sys.path.insert(0,str(HERE));from prepare import METHODS,SEEDS
from prepare_imdb import clean


def validate_predictions(path,ids,labels):
    with path.open() as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(ids):raise ValueError(f'Row count mismatch {path}')
    idx='orig_idx' if 'orig_idx' in rows[0] else 'imdb_idx'
    if not np.array_equal([int(r[idx]) for r in rows],ids):raise ValueError(f'ID mismatch {path}')
    if not np.array_equal([int(r['label']) for r in rows],labels):raise ValueError(f'Label mismatch {path}')
    y=np.asarray(labels);pred=np.asarray([int(r['prediction']) for r in rows]);p=np.asarray([float(r['p_positive']) for r in rows])
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)) or np.any((p!=.5)&(pred!=(p>.5))):raise ValueError('Invalid prediction')
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),
            'positive_f1':float(f1_score(y,pred))},pred


def aggregate(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[r['name']].append(r)
    out=[]
    for name,rr in sorted(grouped.items()):
        s={'name':name,'n':len(rr)}
        for metric in ('accuracy','macro_f1','positive_f1'):
            x=np.asarray([r[metric] for r in rr]);s[metric+'_mean']=float(x.mean());s[metric+'_sd']=float(x.std(ddof=1)) if len(x)>1 else None
        out.append(s)
    return out


def differences(rows):
    keyed={(r['name'],r['seed']):r for r in rows};out=[]
    for name in METHODS:
        if name=='mix_random':continue
        for seed in SEEDS:
            target=keyed.get((name,seed));base=keyed.get(('mix_random',seed))
            if target and base:
                out.append({'name':name,'seed':seed,'accuracy_delta_pp':100*(target['accuracy']-base['accuracy']),
                            'macro_f1_delta_pp':100*(target['macro_f1']-base['macro_f1'])})
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--imdb-source',type=Path);args=ap.parse_args()
    ds,_=read_data();y=np.asarray(ds['test']['label']);ids=np.asarray(ds['test']['orig_idx']);rows=[]
    for name in list(METHODS)+['random','len8']:
        parent=ROOT/'runs' if name in METHODS else REPO/'results/len_forgetting/runs'
        for seed in SEEDS:
            runid=f'{name}__ss{seed}__ts{seed}';directory=parent/runid;f=directory/'metrics.json'
            if not f.exists():continue
            info=json.loads(f.read_text())
            if info['pilot'] or info['split']!='test' or info['name']!=name or info['selection_seed']!=seed or info['training_seed']!=seed:raise ValueError('Bad run metadata')
            vals,_=validate_predictions(directory/'predictions.csv',ids,y)
            for key,value in vals.items():
                if abs(info['metrics'][key]-value)>1e-10:raise ValueError(f'Metric mismatch {runid} {key}')
            selection_dir=ROOT/'selections' if name in METHODS else REPO/'results/len_forgetting/selections'
            sel=json.loads((selection_dir/f'{name}__s{seed}.json').read_text());checksum=sel.pop('sha256')
            if digest(sel)!=checksum or info['selection_sha256']!=checksum:raise ValueError(f'Selection mismatch {runid}')
            rows.append({'name':name,'seed':seed,**vals,'val_best_accuracy':info['best_metric'],'seconds':info['seconds']})
    report={'expected_new_runs':21,'completed_new_runs':sum(r['name'] in METHODS for r in rows),
            'baseline_runs':sum(r['name'] not in METHODS for r in rows),
            'sst2':{'runs':rows,'summary':aggregate(rows),'paired_deltas':differences(rows)},
            'caveat':'Exploratory; repeated SST-2 evaluation, 3 linked seeds, multiple comparisons, OOF proxy uses all pool labels.'}
    external_files=list((ROOT/'imdb_runs').glob('*/metrics.json')) if (ROOT/'imdb_runs').exists() else []
    if external_files:
        if not args.imdb_source:raise ValueError('Specify --imdb-source to validate external predictions')
        manifest=json.loads((ROOT/'imdb_manifest.json').read_text());external=load_from_disk(str(args.imdb_source));ix=manifest['indices']
        labels=[int(external[i]['label']) for i in ix];texts=[clean(external[i]['text']) for i in ix]
        if digest(list(zip(ix,labels,texts)))!=manifest['data_sha256']:raise ValueError('External sample mismatch')
        ext_rows=[]
        source={(r['name'],r['seed']):r for r in rows}
        for file in external_files:
            info=json.loads(file.read_text());name=info['name'];seed=info['selection_seed']
            if (name,seed) not in source or info['imdb_data_sha256']!=manifest['data_sha256']:raise ValueError('External run/source mismatch')
            vals,_=validate_predictions(file.parent/'predictions.csv',ix,labels)
            for k,v in vals.items():
                if abs(v-info['metrics'][k])>1e-10:raise ValueError(f'External metric mismatch {file}')
            ext_rows.append({'name':name,'seed':seed,**vals,'seconds':info['seconds']})
        report['imdb']={'runs':ext_rows,'summary':aggregate(ext_rows),'paired_deltas':differences(ext_rows),'expected_runs':27,'completed_runs':len(ext_rows)}
    (ROOT/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
