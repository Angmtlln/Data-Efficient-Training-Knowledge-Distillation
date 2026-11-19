"""Summarize round-two exported results and subset properties."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
export = root / 'results/svp_round2_results'
out = root / 'results/svp_round2_analysis'
out.mkdir(parents=True, exist_ok=True)
scores = pd.read_parquet(root / 'data/scores.parquet').sort_values('pool_idx').reset_index(drop=True)
if not np.array_equal(scores.pool_idx, np.arange(len(scores))):
    raise ValueError('Scores must be indexed by pool_idx')
results = pd.read_csv(export / 'results/results.csv')
names = ['random_t4', 'svp_kcenter', 'svp_kcenter_cls_norm', 'svp_kcenter_l3', 'svp_uncertainty', 'svp_forget_epoch', 'svp_forget_l3']
rows = []
for method in names:
    for seed in range(3):
        name = f'{method}_s{seed}'
        run = results[results.subset == name]
        if len(run) != 1 or int(run.seed.iloc[0]) != seed:
            raise ValueError(f'Expected one matching result for {name}')
        idx = np.load(export / f'results/subsets/{name}.npy')
        if len(idx) != int(len(scores) * .1) or len(np.unique(idx)) != len(idx) or (idx < 0).any() or (idx >= len(scores)).any():
            raise ValueError(f'Invalid subset {name}')
        sub = scores.iloc[idx]
        rows.append(dict(method=method, seed=seed, val_acc=run.val_acc.iloc[0], val_f1=run.val_f1.iloc[0],
                         test_acc=run.test_acc.iloc[0], test_f1=run.test_f1.iloc[0],
                         best_epoch=run.best_epoch.iloc[0], train_time_s=run.train_time_s.iloc[0],
                         positive_share=sub.label.mean(), label_issue_share=sub.label_issue.mean(),
                         mean_words=sub.n_words.mean(), oof_gold_prob=sub.oof_gold_prob.mean()))
runs = pd.DataFrame(rows)
runs.to_csv(out / 'runs_and_features.csv', index=False)
cols = ['val_acc','val_f1','test_acc','test_f1','train_time_s','positive_share','label_issue_share','mean_words','oof_gold_prob']
summary = runs.groupby('method')[cols].agg(['mean','std','count']).loc[names]
summary.to_csv(out / 'summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
labels = ['Random T4', 'CLS', 'CLS normalized', 'MiniLM-L3', 'Uncertainty', 'Forgetting epoch', 'Forgetting MiniLM-L3']
colors = ['#73869a','#79a1b0','#5a93a5','#218879','#c97769','#be9a67','#a96861']
pos = np.arange(len(names))
for ax, column, title, xlabel, scale in [
    (axes[0], 'test_acc', 'GLUE validation: accuracy', 'Accuracy', 1),
    (axes[1], 'label_issue_share', 'Potential label issues (existing flag)', 'Share, %', 100),
]:
    ax.barh(pos, summary[(column,'mean')].to_numpy()*scale, xerr=summary[(column,'std')].to_numpy()*scale, color=colors, capsize=3)
    ax.set_yticks(pos, labels); ax.invert_yaxis(); ax.set_title(title); ax.set_xlabel(xlabel)
axes[0].set_xlim(0,1); fig.tight_layout(); fig.savefig(out/'comparison.png',dpi=180); plt.close(fig)
print(f'Saved results in {out}')
