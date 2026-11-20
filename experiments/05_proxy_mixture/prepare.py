"""Build matched long-text mixtures from frozen OOF and lexical scores."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];ROOT=REPO/'results/proxy_mixture'
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'))
from common import read_data,digest
N=6534;DOSE=1634;SEEDS=(17,42,101)
METHODS=('mix_random','mix_hard','mix_uncertain','mix_mid','mix_easy','mix_rare','mix_diverse')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stratified_mixture(labels,lengths,p_gold,rare,clusters,seed,n=N,dose=DOSE):
    y=np.asarray(labels);lengths=np.asarray(lengths);p=np.asarray(p_gold);rare=np.asarray(rare);clusters=np.asarray(clusters)
    if not (len(y)==len(lengths)==len(p)==len(rare)==len(clusters)) or not (0<dose<n):raise ValueError('Invalid input shapes/budget')
    if not np.isfinite(p).all() or not np.isfinite(rare).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid scores')
    rng=np.random.default_rng(seed);bins=np.minimum(lengths//4,8);eligible=np.flatnonzero(lengths>=8)
    if len(eligible)<n:raise ValueError('Insufficient long candidates')
    if (clusters[eligible]<0).any():raise ValueError('Missing cluster assignments')
    bg=rng.choice(eligible,n-dose,replace=False)
    avail=np.setdiff1d(eligible,bg,assume_unique=False)
    slots=rng.choice(avail,dose,replace=False)
    h=lambda a:Counter(zip(y[a].tolist(),bins[a].tolist()))
    quota=h(slots);out={m:[] for m in METHODS}
    for key,count in sorted(quota.items()):
        label,length_bin=key;candidates=avail[(y[avail]==label)&(bins[avail]==length_bin)]
        if len(candidates)<count:raise ValueError(f'Insufficient candidates in {key}')
        random_ids=slots[(y[slots]==label)&(bins[slots]==length_bin)]
        out['mix_random'].extend(random_ids.tolist())
        # Lexsort's first key randomly resolves tied scores, reproducibly within each group.
        for name,score in [('mix_hard',p[candidates]),('mix_uncertain',np.abs(p[candidates]-.5)),
                           ('mix_mid',np.abs(p[candidates]-np.median(p[candidates]))),
                           ('mix_easy',-p[candidates]),('mix_rare',-rare[candidates])]:
            ranked=np.lexsort((rng.random(len(candidates)),score))
            out[name].extend(candidates[ranked[:count]].tolist())
        # Round-robin over non-empty clusters, without replacement and with seeded tie order.
        buckets={}
        for cluster_id in np.unique(clusters[candidates]):
            group=candidates[clusters[candidates]==cluster_id];buckets[int(cluster_id)]=rng.permutation(group).tolist()
        order=rng.permutation(np.asarray(sorted(buckets),dtype=int)).tolist();selected=[]
        while len(selected)<count:
            for cluster_id in order:
                if buckets[cluster_id]:selected.append(buckets[cluster_id].pop())
                if len(selected)==count:break
        out['mix_diverse'].extend(selected)
    result={}
    for name in METHODS:
        target=np.asarray(out[name],dtype=np.int64);all_ids=np.concatenate([bg,target])
        if len(all_ids)!=n or len(np.unique(all_ids))!=n or h(target)!=quota or np.any(lengths[all_ids]<8):
            raise ValueError(f'Invalid selection {name}')
        result[name]=np.sort(all_ids)
    return result,bg,quota


def main():
    ds,meta=read_data(); pool=ds['pool'];y=np.asarray(pool['label']);lens=np.asarray([len(s.split()) for s in pool['sentence']]);
    proxy_meta=json.loads((ROOT/'proxy/metadata.json').read_text())
    if proxy_meta['pool_sha256']!=meta['splits']['pool']['sha256']:raise ValueError('Proxy pool hash mismatch')
    lexical_path=ROOT/'proxy/lexical_features.npz'
    if sha(lexical_path)!=proxy_meta['lexical_sha256']:raise ValueError('Lexical hash mismatch')
    lexical=np.load(lexical_path);clusters=lexical['cluster'];rare=lexical['rare_idf']
    out=ROOT/'selections';out.mkdir(parents=True,exist_ok=True);audit=[]
    for seed in SEEDS:
        score_path=ROOT/'proxy'/f'gold_probability_s{seed}.npz'
        if sha(score_path)!=next(x['sha256'] for x in proxy_meta['seeds'] if x['seed']==seed):raise ValueError('Score hash mismatch')
        p=np.load(score_path)['p_gold'];sets,bg,quota=stratified_mixture(y,lens,p,rare,clusters,seed)
        for name,arr in sets.items():
            manifest={'name':name,'selection_seed':seed,'indices':arr.tolist(),'n':N,'ratio':.1,
                'pool_sha256':meta['splits']['pool']['sha256'],
                'provenance':{'selection':'4900 shared random long background + 1634 matched class/length slots',
                    'oof_score_seed':seed,'oof_score_sha256':sha(score_path),'lexical_sha256':sha(lexical_path),
                    'labels':'entire pool available to OOF proxy and selector','long_definition':'len(text.split()) >= 8'}}
            manifest['sha256']=digest(manifest);(out/f'{name}__s{seed}.json').write_text(json.dumps(manifest,indent=2)+'\n')
            target=np.setdiff1d(arr,bg,assume_unique=False)
            audit.append({'name':name,'seed':seed,'n':len(arr),'background_n':len(bg),'target_n':len(target),
                'positive':int(y[arr].sum()),'mean_words':float(lens[arr].mean()),
                'target_oof_gold_mean':float(p[target].mean()),'target_oof_wrong_fraction':float((p[target]<.5).mean()),
                'target_rare_idf_mean':float(rare[target].mean()),'target_cluster_count':int(len(np.unique(clusters[target]))),
                'target_cluster_largest_share':float(np.max(np.bincount(clusters[target]))/len(target)),
                'target_strata':[{'label':int(a),'length_bin':int(b),'count':int(v)} for (a,b),v in sorted(quota.items())]})
    (ROOT/'selection_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    for r in audit:print(r['seed'],r['name'],'pos',r['positive'],'gold',round(r['target_oof_gold_mean'],3),
                          'proxy_wrong',round(r['target_oof_wrong_fraction'],3),'clusters',r['target_cluster_count'],flush=True)

if __name__=='__main__':main()
