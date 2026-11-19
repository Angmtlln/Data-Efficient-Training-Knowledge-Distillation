"""Validate and summarize the exported gradient-selection experiments."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'results/gradient_round3_results'
OUT = ROOT / 'results/gradient_round3_analysis'
OUT.mkdir(parents=True, exist_ok=True)
scores = pd.read_parquet(ROOT / 'data/scores.parquet').sort_values('pool_idx').reset_index(drop=True)
if not np.array_equal(scores.pool_idx, np.arange(len(scores))):
    raise ValueError('pool indices do not align')
selection = np.load(EXPORT / 'data/gradient_selection_val_idx.npy')
evaluation = np.load(EXPORT / 'data/gradient_epoch_val_idx.npy')
if len(selection) != 1000 or len(evaluation) != 1000 or len(np.unique(np.r_[selection, evaluation])) != 2000 or min(selection.min(),evaluation.min()) != 0 or max(selection.max(),evaluation.max()) != 1999:
    raise ValueError('invalid validation partition')
results = pd.read_csv(EXPORT / 'results/results.csv')
rows=[]
for method in ['grad_random','grad_len8','grad_glister','grad_craig']:
    for seed in range(3):
        name=f'{method}_s{seed}'
        r=results[(results.subset == name)&(results.seed==seed)&(results.tag=='gradient_round3')]
        if len(r)!=1 or int(r.n_train.iloc[0]) != int(len(scores)*.1):raise ValueError(f'invalid run {name}')
        idx=np.load(EXPORT/f'results/subsets/{name}.npy')
        if len(idx)!=int(len(scores)*.1) or len(np.unique(idx))!=len(idx) or np.any((idx<0)|(idx>=len(scores))):raise ValueError(f'invalid subset {name}')
        subset=scores.iloc[idx]
        weights=np.load(EXPORT/f'results/subsets/{name}_weights.npy') if method=='grad_craig' else np.ones(len(idx))
        if weights.shape!=(len(idx),) or not np.isfinite(weights).all() or (weights<0).any() or not np.isclose(weights.mean(),1):raise ValueError(f'invalid weights {name}')
        proxy=np.load(ROOT/'results/svp_results/results/subsets/random_s0.npy')
        rows.append(dict(method=method, seed=seed, **{k:r.iloc[0][k] for k in ['val_acc','val_f1','test_acc','test_f1','best_epoch','train_time_s']},
                         positive_share=subset.label.mean(),weighted_positive_share=np.average(subset.label,weights=weights),
                         label_issue_share=subset.label_issue.mean(),weighted_label_issue_share=np.average(subset.label_issue,weights=weights),
                         mean_words=subset.n_words.mean(),zero_weights=int((weights==0).sum()),weight_max=weights.max(),
                         weight_effective_n=weights.sum()**2/(weights@weights),proxy_overlap=len(np.intersect1d(idx,proxy))))
full=results[(results.subset=='full')&(results.seed==0)&(results.tag=='gradient_round3')]
if len(full)!=1 or int(full.n_train.iloc[0]) != len(scores):raise ValueError('full run missing')
runs=pd.DataFrame(rows);runs.to_csv(OUT/'runs_and_features.csv',index=False)
metrics=['val_acc','val_f1','test_acc','test_f1','train_time_s','positive_share','label_issue_share','weighted_label_issue_share','weight_effective_n']
runs.groupby('method')[metrics].agg(['mean','std','count']).to_csv(OUT/'summary.csv')
full.to_csv(OUT/'full.csv',index=False)
proxy=json.loads((EXPORT/'data/gradient_proxy_info.json').read_text())
info={'proxy':proxy,'glister':json.loads((EXPORT/'data/grad_glister_info.json').read_text()),'craig':[json.loads((EXPORT/f'data/grad_craig_s{i}_info.json').read_text()) for i in range(3)]}
(OUT/'timings.json').write_text(json.dumps(info,indent=2))
print('Saved analysis to',OUT)
