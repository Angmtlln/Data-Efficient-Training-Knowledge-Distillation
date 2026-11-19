"""Class-stratified gradient ranking and gradient coverage within length-filtered pool.

These are adaptations of GLISTER/CRAIG, not implementations of their full algorithms.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
from src.config import DATA_DIR, SUBSETS_DIR
from src.selection.gradient_coreset import sparse_facility


def balanced_topk(scores, labels, budget):
    """Top scores within each binary class; quotas match the candidate distribution."""
    scores=np.asarray(scores); labels=np.asarray(labels)
    if scores.ndim!=1 or labels.shape!=scores.shape or not np.isfinite(scores).all() or set(np.unique(labels)) != {0,1} or not 0<budget<=len(scores):
        raise ValueError('invalid scores, binary labels or budget')
    # Largest remainder gives an exact total without using randomness to change ties.
    ideal=np.bincount(labels.astype(int), minlength=2)*budget/len(labels)
    quota=np.floor(ideal).astype(int)
    for j in np.argsort(-(ideal-quota),kind='stable')[:budget-quota.sum()]: quota[j]+=1
    parts=[]
    for cls in (0,1):
        rows=np.flatnonzero(labels==cls)
        parts.append(rows[np.argsort(-scores[rows],kind='stable')[:quota[cls]]])
    return np.concatenate(parts)


def facility_projected(x,budget,seed,dimensions=64,k=32):
    import faiss
    faiss.omp_set_num_threads(4)
    rng=np.random.default_rng(seed); projection=rng.choice(np.array([-1,1],dtype=np.float32),size=(x.shape[1],dimensions))/np.sqrt(dimensions)
    z=np.ascontiguousarray(x @ projection,dtype=np.float32); n=len(z)
    ix=faiss.IndexHNSWFlat(dimensions,32); ix.hnsw.efConstruction=80;ix.hnsw.efSearch=80;ix.add(z)
    d,nn=ix.search(z,min(k,n))
    if (nn<0).any():raise RuntimeError('incomplete graph')
    for i in range(n):
        found=np.flatnonzero(nn[i]==i)
        if len(found):
            j=found[0]; nn[i,0],nn[i,j]=nn[i,j],nn[i,0];d[i,0],d[i,j]=d[i,j],d[i,0]
        else:nn[i,0]=i
        d[i,0]=0
    positives=d[d>0]; scale=float(np.median(positives)) if len(positives) else 1.
    similarity=np.exp(-np.maximum(d,0)/max(scale,1e-12)).astype(np.float32)
    chosen=sparse_facility(nn,similarity,budget)
    nearest=faiss.IndexHNSWFlat(dimensions,32);nearest.hnsw.efConstruction=80;nearest.hnsw.efSearch=80;nearest.add(z[chosen])
    _,assignment=nearest.search(z,1)
    if (assignment<0).any():raise ValueError('assignment failed')
    weights=np.bincount(assignment[:,0],minlength=budget).astype(np.float32)
    if weights.sum()!=n:raise ValueError('weights fail to cover candidates')
    weights/=weights.mean()
    return chosen,weights


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2]);p.add_argument('--budget',type=float,default=.1)
    args=p.parse_args(); start=time.perf_counter()
    df=pd.read_parquet(DATA_DIR/'scores.parquet').sort_values('pool_idx').reset_index(drop=True)
    x=np.load(DATA_DIR/'gradient_pool.npy',mmap_mode='r'); v=np.load(DATA_DIR/'gradient_val_mean.npy')
    if len(x)!=len(df) or not np.array_equal(df.pool_idx,np.arange(len(df))) or x.ndim!=2 or v.shape!=(x.shape[1],) or not np.isfinite(x).all() or not np.isfinite(v).all():raise ValueError('unaligned or invalid pool gradients')
    budget=int(len(df)*args.budget)
    if not 0<budget<=len(df):raise ValueError('invalid budget')
    candidate_sets={'all':np.arange(len(df)), 'len8':np.flatnonzero(df.n_words.to_numpy()>=8)}
    SUBSETS_DIR.mkdir(parents=True,exist_ok=True)
    scores=np.asarray(x @ v)
    np.save(DATA_DIR/'round4_glister_scores.npy',scores)
    info={'pool_n':len(df),'budget':budget,'candidate_len8_n':len(candidate_sets['len8']), 'proxy_reused_for_seeds':True,'methods':{}}
    for kind,candidates in candidate_sets.items():
        t=time.perf_counter(); local=balanced_topk(scores[candidates],df.label.to_numpy()[candidates],budget);chosen=candidates[local]
        name='r4_glister_bal' if kind=='all' else 'r4_glister_bal_len8'
        for seed in args.seeds:np.save(SUBSETS_DIR/f'{name}_s{seed}.npy',chosen)
        info['methods'][name]={'selection_seconds':time.perf_counter()-t,'positive_count':int(df.label.iloc[chosen].sum())}
        print(name,len(chosen),'positive',df.label.iloc[chosen].mean(),flush=True)
    candidates=candidate_sets['len8']
    for seed in args.seeds:
        t=time.perf_counter();chosen,weights=facility_projected(x[candidates],budget,seed);indices=candidates[chosen]
        np.save(SUBSETS_DIR/f'r4_craig_len8_s{seed}.npy',indices)
        np.save(SUBSETS_DIR/f'r4_craig_len8_s{seed}_weights.npy',weights)
        np.save(SUBSETS_DIR/f'r4_craig_len8_unweighted_s{seed}.npy',indices)
        info['methods'][f'r4_craig_len8_s{seed}']={'selection_seconds':time.perf_counter()-t,'zero_weights':int((weights==0).sum()),'effective_n':float(weights.sum()**2/(weights@weights))}
        print('r4_craig_len8',seed,len(indices),'positive',df.label.iloc[indices].mean(),flush=True)
    info['total_seconds_including_loading']=time.perf_counter()-start
    (DATA_DIR/'round4_selection_info.json').write_text(json.dumps(info,indent=2));print(info,flush=True)

if __name__=='__main__':main()
