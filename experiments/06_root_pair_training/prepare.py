"""Map original SST train trees to the frozen GLUE pool; prepare paired ablations."""
import argparse, csv, hashlib, json, re, sys, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'code'))
from common import ROOT, read_data, dump, digest

SOURCE='https://huggingface.co/datasets/stanfordnlp/sst/resolve/refs%2Fconvert%2Fparquet/ptb/train/0000.parquet'
SOURCE_SHA='4e55a3cf2c4de55a121b51af013250841c92f0abf1aae93e289b09df01814144'
def norm(s):
    s=unicodedata.normalize('NFKC',s).lower().replace('-lrb-','(').replace('-rrb-',')').replace('\\/','/').replace('\\*','*')
    return ' '.join(s.split())
def parse(s):
    tok=re.findall(r'\(|\)|[^\s()]+',s); i=0; nodes=[]
    def visit():
        nonlocal i
        assert tok[i]=='('; label=int(tok[i+1]); i+=2; words=[]
        while tok[i]!=')':
            if tok[i]=='(': words.extend(visit()[1])
            else: words.append(tok[i]); i+=1
        i+=1; node=(label,words); nodes.append(node); return node
    root=visit(); assert i==len(tok); return root,nodes

def prepare(source, seeds, requested):
    import pandas as pd
    raw=source.read_bytes(); assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA,'Unexpected source bytes'
    ds,meta=read_data(); pool=ds['pool']; texts=pool['sentence']; labels=np.array(pool['label']); lookup={}
    for i,(t,y) in enumerate(zip(texts,labels)): lookup.setdefault((norm(t),int(y)),[]).append(i)
    excluded={norm(t) for split in ['val','test'] for t in ds[split]['sentence']}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'data/tokenizer'),local_files_only=True)
    roots=set(); pairs=[]; source_roots={}; excluded_roots=0
    for tid,s in enumerate(pd.read_parquet(source)['ptb_tree']):
        (rlabel,rwords),nodes=parse(s)
        if rlabel==2: continue
        ry=int(rlabel>2); rt=norm(' '.join(rwords)); ids=lookup.get((rt,ry),[])
        if not ids: continue
        if rt in excluded: excluded_roots+=1; continue
        root=min(ids); roots.add(root); source_roots[root]=tid
        for nl,nw in nodes[:-1]:
            if nl==2 or int(nl>2)==ry: continue
            nt=norm(' '.join(nw)); child=lookup.get((nt,int(nl>2)),[])
            if not child or nt in excluded or len(nw)<8 or len(nw)/len(rwords)<.5: continue
            cid=min(child)
            if cid==root: continue
            rw=re.findall(r"[a-z]+(?:'[a-z]+)?",rt); cw=re.findall(r"[a-z]+(?:'[a-z]+)?",nt)
            added=Counter(rw)-Counter(cw)
            if sum(added.values())<2: continue
            if not set(added)&{'but','although','though','however','yet','despite','not',"n't",'never'}: continue
            if not set(rw)&{'but','although','though','however','yet','despite','not',"n't",'never'}: continue
            if cw and cw[-1] in {'but','although','though','however','despite','and','or'}: continue
            if max(len(tokenizer(texts[k])['input_ids']) for k in [root,cid])>64: continue
            pairs.append(dict(root=root,child=cid,tree_id=tid,root_label=ry,child_label=int(nl>2),root_words=len(rwords),child_words=len(nw),root_text=texts[root],child_text=texts[cid]))
    pairs=[p for p in pairs if p['child'] not in roots]
    # Longest opposite-label child per root; then disjoint children across roots.
    chosen=[]; used=set()
    for p in sorted(pairs,key=lambda p:(-p['child_words'],-p['root_words'],p['root'],p['child'])):
        if p['root'] in used or p['child'] in used: continue
        chosen.append(p); used.update([p['root'],p['child']])
    ROOT.joinpath('reports').mkdir(exist_ok=True)
    for name,rows in [('candidates',pairs),('eligible_pairs',chosen)]:
        if rows:
            with (ROOT/'reports'/f'{name}.csv').open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(ROOT/'reports/source.json',dict(url=SOURCE,sha256=SOURCE_SHA,trees=len(pd.read_parquet(source)),roots=len(roots),candidate_pairs=len(pairs),disjoint_pairs=len(chosen),excluded_roots=excluded_roots))
    if len(roots)<6534: raise ValueError(f'Only {len(roots)} unique eligible roots; cannot meet 6534 budget without changing protocol')
    review=json.loads((ROOT/'pair_review.json').read_text())
    approved={(r['root'],r['child']) for r in review if r['accepted']}
    chosen=[r for r in chosen if (r['root'],r['child']) in approved]
    assert len(chosen)==len(approved),'Review no longer matches candidates'
    n=min(requested,len(chosen)); assert n>0
    audits=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); pp=[chosen[i] for i in rng.permutation(len(chosen))[:n]]
        required={p['root'] for p in pp}; base=sorted(required|set(rng.choice(sorted(roots-required),6534-len(required),replace=False).tolist()))
        removable=set(base)-required; removed=[]
        # Every replaced root matches child label; nearest available length.
        for p in pp:
            cand=[i for i in removable if labels[i]==p['child_label']]
            if not cand: raise ValueError('No same-label replacement')
            best=min(cand,key=lambda i:(abs(len(texts[i].split())-p['child_words']),i));removed.append(best);removable.remove(best)
        mixed=sorted((set(base)-set(removed))|{p['child'] for p in pp})
        assert len(mixed)==len(set(mixed))==6534
        assert np.bincount(labels[base]).tolist()==np.bincount(labels[mixed]).tolist()
        for name,ids,mode,weight in [('roots',base,'independent',0),('mixed_independent',mixed,'independent',0),('mixed_paired',mixed,'paired',0),('mixed_rank',mixed,'paired',.2)]:
            value=dict(name=name,selection_seed=seed,indices=ids,pairs=[[p['root'],p['child']] for p in pp] if name!='roots' else [],batch_mode=mode,rank_weight=weight,rank_margin=.2,pool_sha256=meta['splits']['pool']['sha256'],source_sha256=SOURCE_SHA,review_sha256=digest(review))
            value['sha256']=digest(value);dump(ROOT/'selections'/f'{name}__s{seed}.json',value)
            audits.append(dict(method=name,seed=seed,n=len(ids),pairs=n if name!='roots' else 0,positive=int(labels[ids].sum()),mean_words=float(np.mean([len(texts[i].split()) for i in ids]))))
        dump(ROOT/'reports'/f'replacements_s{seed}.json',[dict(**p,removed_index=i,removed_text=texts[i]) for p,i in zip(pp,removed)])
    pd.DataFrame(audits).to_csv(ROOT/'reports/selection_audit.csv',index=False)
    print(json.dumps(dict(roots=len(roots),eligible_pairs=len(chosen),selected_pairs=n,runs=len(audits)),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--trees',type=Path,default=ROOT/'inputs/sst_train.parquet');p.add_argument('--seeds',type=int,nargs='+',default=[1103,1201,1301]);p.add_argument('--pairs',type=int,default=131);a=p.parse_args();prepare(a.trees,a.seeds,a.pairs)
