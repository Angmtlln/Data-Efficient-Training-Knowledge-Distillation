"""Validate archived predictions and summarize any completed, non-pilot runs."""
import csv
import json
from pathlib import Path
import sys
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'))
from common import digest,read_data


def main():
    root=REPO/'results/len_forgetting';ds,_=read_data();test=ds['test']
    y=np.asarray(test['label']); ids=np.asarray(test['orig_idx']); groups=defaultdict(list); pairs=defaultdict(dict)
    predictions={}
    for file in sorted((root/'runs').glob('*/metrics.json')):
        item=json.loads(file.read_text())
        if item['pilot'] or item['split']!='test':raise ValueError(f'Pilot/non-test in {file}')
        with (file.parent/'predictions.csv').open() as f: rows=list(csv.DictReader(f))
        if len(rows)!=len(y):raise ValueError(f'Wrong prediction count in {file}')
        if not np.array_equal([int(r['orig_idx']) for r in rows],ids):raise ValueError(f'ID order mismatch in {file}')
        if not np.array_equal([int(r['label']) for r in rows],y):raise ValueError(f'Labels mismatch in {file}')
        predicted=np.array([int(r['prediction']) for r in rows]);probs=np.array([float(r['p_positive']) for r in rows])
        if not np.isfinite(probs).all() or ((probs<0)|(probs>1)).any(): raise ValueError('Invalid probabilities')
        # At a floating point tie np.argmax chooses class zero, so skip exact ties.
        if np.any((probs!=0.5) & (predicted != (probs>0.5))):raise ValueError('Prediction probability mismatch')
        values={'accuracy':float(accuracy_score(y,predicted)), 'macro_f1':float(f1_score(y,predicted,average='macro')),
                'positive_f1':float(f1_score(y,predicted))}
        for key, value in values.items():
            if abs(item['metrics'][key]-value)>1e-10: raise ValueError(f'Metric mismatch {key} {file}')
        name=item['name'];seed=item['selection_seed'];path=root/'selections'/f'{name}__s{seed}.json'
        selection=json.loads(path.read_text());sha=selection.pop('sha256')
        if digest(selection)!=sha or sha!=item['selection_sha256']:raise ValueError('Selection mismatch')
        row={'name':name,'selection_seed':seed,'training_seed':item['training_seed'],**values,
             'val_best_accuracy':item['best_metric'],'seconds':item['seconds'],'device':item['device']}
        groups[name].append(row);pairs[(seed,item['training_seed'])][name]=row
        predictions[(seed,item['training_seed'],name)]=predicted
    output=[]
    for name,rows in sorted(groups.items()):
        d={'name':name,'n':len(rows)}
        for key in ['accuracy','macro_f1','positive_f1','val_best_accuracy','seconds']:
            x=np.asarray([r[key] for r in rows]);d[key+'_mean']=float(x.mean());d[key+'_sd']=float(x.std(ddof=1)) if len(x)>1 else None
        output.append(d)
    pair_rows=[]
    for (seed,train_seed),v in sorted(pairs.items()):
        if 'len8_forgetting' in v and 'len8_matched_random' in v:
            control=predictions[(seed,train_seed,'len8_matched_random')]
            treatment=predictions[(seed,train_seed,'len8_forgetting')]
            pair_rows.append({'selection_seed':seed,'training_seed':train_seed,
                'accuracy_delta_pp':100*(v['len8_forgetting']['accuracy']-v['len8_matched_random']['accuracy']),
                'macro_f1_delta_pp':100*(v['len8_forgetting']['macro_f1']-v['len8_matched_random']['macro_f1']),
                'different_predictions':int(np.sum(control!=treatment)),
                'control_correct_only':int(np.sum((control==y)&(treatment!=y))),
                'treatment_correct_only':int(np.sum((control!=y)&(treatment==y)))})
    report={'n_runs':sum(len(v) for v in groups.values()),'expected_runs':12,'summary':output,
            'paired_comparisons':pair_rows,
            'caveat':'Exploratory: one training seed per selection seed; fixed 872-row GLUE validation reused in prior experiments.'}
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
