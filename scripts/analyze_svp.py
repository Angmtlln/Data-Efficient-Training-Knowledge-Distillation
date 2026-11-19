"""Summarize the exported Colab SVP run without changing the original results."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'results' / 'svp_results'
OUT = ROOT / 'results' / 'svp_analysis'
OUT.mkdir(parents=True, exist_ok=True)
scores = pd.read_parquet(ROOT / 'data/scores.parquet').sort_values('pool_idx').reset_index(drop=True)
proxy = pd.read_parquet(EXPORT / 'data/svp_proxy_scores.parquet').sort_values('pool_idx')
if not np.array_equal(scores.pool_idx, np.arange(len(scores))) or not np.array_equal(scores.pool_idx, proxy.pool_idx):
    raise ValueError('Scores and pool indices do not align')
scores['proxy_gold_prob'] = proxy.gold_prob_cheap.to_numpy()
p = scores.proxy_gold_prob.clip(1e-12, 1 - 1e-12)
scores['proxy_entropy'] = -(p * np.log(p) + (1 - p) * np.log1p(-p))
rows = []
for name in ('random', 'svp_kcenter', 'svp_uncertainty'):
    for seed in range(3):
        idx = np.load(EXPORT / f'results/subsets/{name}_s{seed}.npy')
        if len(idx) != int(len(scores) * 0.1) or len(np.unique(idx)) != len(idx) or (idx < 0).any() or (idx >= len(scores)).any():
            raise ValueError(f'invalid subset {name} {seed}')
        sub = scores.iloc[idx]
        rows.append(dict(strategy=name, seed=seed, n=len(idx), positive_share=sub.label.mean(),
                         words_mean=sub.n_words.mean(), label_issue_share=sub.label_issue.mean(),
                         oof_gold_prob_mean=sub.oof_gold_prob.mean(),
                         proxy_correct_share=(sub.proxy_gold_prob > .5).mean(),
                         proxy_entropy_mean=sub.proxy_entropy.mean()))
features = pd.DataFrame(rows)
features.to_csv(OUT / 'subset_features.csv', index=False)
results = pd.read_csv(EXPORT / 'results/results.csv')
results = results[results.subset.str.match(r'^(random|kcenter|svp_kcenter|svp_uncertainty)_s[012]$')].copy()
results['strategy'] = results.subset.str.replace(r'_s\d+$', '', regex=True)
results.to_csv(OUT / 'runs.csv', index=False)
summary = results.groupby('strategy')[['val_acc', 'val_f1', 'test_acc', 'test_f1', 'train_time_s']].agg(['mean', 'std', 'count'])
summary.to_csv(OUT / 'summary.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
order = ['random', 'kcenter', 'svp_kcenter', 'svp_uncertainty']
means = summary.loc[order, ('test_acc', 'mean')]
std = summary.loc[order, ('test_acc', 'std')]
axes[0].bar(range(4), means, yerr=std, capsize=4, color=['#8e9aaf','#5a7898','#459989','#c87765'])
axes[0].set_xticks(range(4), ['Random', 'K-center\n(MiniLM)', 'SVP\nk-center', 'SVP\nuncertainty'])
axes[0].set_ylim(0, 1); axes[0].set_ylabel('Accuracy'); axes[0].set_title('GLUE validation (n = 872), mean ± SD')
order2 = ['random', 'svp_kcenter', 'svp_uncertainty']
f = features.groupby('strategy').label_issue_share.agg(['mean','std']).loc[order2] * 100
axes[1].bar(range(3), f['mean'], yerr=f['std'], capsize=4, color=['#8e9aaf','#459989','#c87765'])
axes[1].set_xticks(range(3), ['Random', 'SVP k-center', 'SVP uncertainty'])
axes[1].set_ylabel('Potential label issues (%)'); axes[1].set_title('Existing label-issue flag, mean ± SD')
fig.tight_layout(); fig.savefig(OUT / 'comparison.png', dpi=180); plt.close(fig)
print('Saved', OUT)
