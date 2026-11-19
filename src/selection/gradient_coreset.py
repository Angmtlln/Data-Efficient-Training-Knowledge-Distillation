"""Static GLISTER-inspired ranking and sparse CRAIG-inspired facility location."""
import argparse
import heapq
import json
import time
from pathlib import Path
import numpy as np
from src.config import DATA_DIR, SUBSETS_DIR


def glister_static(gradients, val_gradient, budget):
    """First-order estimate of validation improvement at one fixed proxy checkpoint."""
    return np.argsort(-(gradients @ val_gradient), kind='stable')[:budget]


def sparse_facility(neighbors, similarities, budget):
    """Lazy greedy maximize sum_i max_{j selected} similarity(i,j) on a sparse graph."""
    n, k = neighbors.shape
    if similarities.shape != (n,k) or not 0 < budget <= n: raise ValueError('invalid graph')
    covered = np.zeros(n,dtype=np.float32)
    heap = [(-float(similarities[i].sum()),i,0) for i in range(n)]
    heapq.heapify(heap); selected=[]
    while len(selected) < budget:
        _, candidate, stamp = heapq.heappop(heap)
        if stamp == len(selected):
            selected.append(candidate)
            idx=neighbors[candidate]; covered[idx]=np.maximum(covered[idx],similarities[candidate])
        else:
            idx=neighbors[candidate]
            gain=np.maximum(similarities[candidate]-covered[idx],0).sum()
            heapq.heappush(heap,(-float(gain),candidate,len(selected)))
    return np.asarray(selected,dtype=np.int64)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--method',choices=['glister','craig'],required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]); parser.add_argument('--budget',type=float,default=.1)
    parser.add_argument('--projection-dim',type=int,default=64); parser.add_argument('--neighbors',type=int,default=32)
    args=parser.parse_args(); x=np.load(DATA_DIR/'gradient_pool.npy',mmap_mode='r'); n=len(x); budget=int(n*args.budget)
    if x.ndim != 2 or not 0<budget<=n or not np.isfinite(x).all(): raise ValueError('invalid gradients')
    SUBSETS_DIR.mkdir(parents=True,exist_ok=True)
    if args.method=='glister':
        v=np.load(DATA_DIR/'gradient_val_mean.npy')
        if v.shape != (x.shape[1],) or not np.isfinite(v).all(): raise ValueError('invalid selection gradients')
        t=time.perf_counter(); idx=glister_static(x,v,budget); elapsed=time.perf_counter()-t
        for seed in args.seeds: np.save(SUBSETS_DIR/f'grad_glister_s{seed}.npy',idx)
        info={'method':'static_first_order_glister_inspired','selection_seconds':elapsed,'same_subset_all_seeds':True}
    else:
        import faiss
        faiss.omp_set_num_threads(4)
        for seed in args.seeds:
            t=time.perf_counter(); rng=np.random.default_rng(seed)
            proj=rng.choice(np.array([-1,1],dtype=np.float32),size=(x.shape[1],args.projection_dim))/np.sqrt(args.projection_dim)
            z=np.ascontiguousarray(x @ proj,dtype=np.float32)
            index=faiss.IndexHNSWFlat(z.shape[1],32); index.hnsw.efConstruction=80; index.hnsw.efSearch=80; index.add(z)
            distances,neighbors=index.search(z,args.neighbors)
            if (neighbors<0).any(): raise RuntimeError('nearest neighbor graph incomplete')
            # Self links guarantee that each selected facility covers itself.
            for i in range(n):
                positions = np.flatnonzero(neighbors[i] == i)
                if len(positions):
                    j = positions[0]
                    neighbors[i,0], neighbors[i,j] = neighbors[i,j], neighbors[i,0]
                    distances[i,0], distances[i,j] = distances[i,j], distances[i,0]
                else:
                    neighbors[i,0] = i
                distances[i,0] = 0
            positive=distances[distances>0]; scale=float(np.median(positive)) if len(positive) else 1.
            sim=np.exp(-np.maximum(distances,0)/max(scale,1e-12)).astype(np.float32)
            idx=sparse_facility(neighbors,sim,budget)
            # Approximate nearest selected facility supplies CRAIG-style coreset weights.
            selected_index=faiss.IndexHNSWFlat(z.shape[1],32); selected_index.hnsw.efConstruction=80; selected_index.hnsw.efSearch=80; selected_index.add(z[idx])
            _, assignment=selected_index.search(z,1)
            weights=np.bincount(assignment[:,0],minlength=budget).astype(np.float32)
            if weights.sum()!=n: raise ValueError('weights do not cover the pool')
            weights /= weights.mean()
            np.save(SUBSETS_DIR/f'grad_craig_s{seed}.npy',idx)
            np.save(SUBSETS_DIR/f'grad_craig_s{seed}_weights.npy',weights)
            info={'method':'sparse_projected_craig_inspired','seed':seed,'projection_dim':args.projection_dim,'neighbors':args.neighbors,'selection_seconds':time.perf_counter()-t,'zero_weight_count':int((weights==0).sum())}
            (DATA_DIR/f'grad_craig_s{seed}_info.json').write_text(json.dumps(info,indent=2)); print(info,flush=True)
    if args.method=='glister':
        (DATA_DIR/'grad_glister_info.json').write_text(json.dumps(info,indent=2)); print(info,flush=True)

if __name__=='__main__': main()
