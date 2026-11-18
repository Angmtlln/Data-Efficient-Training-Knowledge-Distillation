"""CPU-only selection; no ModernBERT training or test-based selection."""
import re
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from common import ROOT, read_data, ranked, manifest, dump, environment


def main():
    ds, meta = read_data()
    texts, y = list(ds['pool']['sentence']), np.asarray(ds['pool']['label'])
    n = 6534
    start = time.time()
    words = [re.findall(r"\b\w+\b", t.lower()) for t in texts]
    from collections import Counter
    df = Counter(w for row in words for w in set(row))
    lexical = np.asarray([len(set(row))/max(1,len(row)) for row in words])
    rare = np.asarray([np.mean([1/df[w] for w in row]) if row else 0 for row in words])
    seen, unique = set(), []
    for i,t in enumerate(texts):
        key = t.strip()
        if key not in seen: unique.append(i); seen.add(key)
    selections = []
    for seed in [17,42,101]:
        rng = np.random.default_rng(seed)
        for name, scores, eligible in [('random',rng.random(len(y)),None),
                ('lexical_diversity',lexical,None),('rare_words',rare,None),
                ('dedup_exact',rng.random(len(y)),unique)]:
            ids = ranked(y,scores,n,seed,eligible)
            manifest(name,ids,seed,meta,dict(selector=name,cpu_seconds=time.time()-start,
                label_budget='All pool labels accessible; final model trains on 6534 rows'))
            selections.append(dict(method=name,seed=seed,n=len(ids),mean_words=np.mean([len(words[i]) for i in ids]),positive_share=float(y[ids].mean())))
        # Fold-heldout transitions: each scored row never trains its scoring proxy.
        scores_path = ROOT / 'data' / f'forgetting_s{seed}.npz'
        forgetting, correct_count = np.zeros(len(y),dtype=int), np.zeros(len(y),dtype=int)
        folds = StratifiedKFold(3,shuffle=True,random_state=seed)
        fold_metrics = []
        for fold,(train,held) in enumerate(folds.split(np.zeros(len(y)), y)):
            vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True)
            x=vectorizer.fit_transform([texts[i] for i in train])
            z=vectorizer.transform([texts[i] for i in held])
            model=SGDClassifier(loss='log_loss',alpha=1e-5,random_state=seed+fold,average=True)
            prev=None
            order_rng=np.random.default_rng(seed+fold)
            for epoch in range(8):
                order=order_rng.permutation(len(train))
                model.partial_fit(x[order],y[train][order],classes=np.array([0,1]))
                correct=model.predict(z)==y[held]
                correct_count[held]+=correct
                if prev is not None: forgetting[held]+=(prev & ~correct)
                prev=correct
                fold_metrics.append(dict(fold=fold,epoch=epoch+1,heldout_accuracy=float(correct.mean())))
            print(f'forgetting seed={seed} fold={fold} finished',flush=True)
        np.savez_compressed(scores_path,forgetting=forgetting,correct_count=correct_count)
        pd.DataFrame(fold_metrics).to_csv(ROOT / 'reports'/f'proxy_history_s{seed}.csv',index=False)
        ids=ranked(y,forgetting,n,seed)
        manifest('heldout_forgetting',ids,seed,meta,dict(proxy='TF-IDF + SGD log-loss',
            folds=3,epochs=8,score='heldout correct-to-incorrect transitions',
            cpu_seconds=time.time()-start,unique_scores=np.unique(forgetting).tolist(),
            note='Not Dataset Cartography and not a faithful SVP reproduction; no KD.'))
        selections.append(dict(method='heldout_forgetting',seed=seed,n=len(ids),mean_words=np.mean([len(words[i]) for i in ids]),positive_share=float(y[ids].mean())))
    pd.DataFrame(selections).to_csv(ROOT / 'reports/selection_summary.csv',index=False)
    dump(ROOT / 'reports/preparation.json',dict(status='complete',seconds=time.time()-start,
         exact_duplicate_rows=len(texts)-len(unique),environment=environment(),gpu_used=False))
    print('CPU preparation complete',flush=True)

if __name__=='__main__': main()
