"""Real frozen MiniLM embeddings + fitted logistic head; raw gradient alignment.

This is a cheap head-gradient experiment, not a reproduction of LESS.
Per-example cosine is deliberately NOT used: in binary logistic regression it
cancels residual magnitude and largely removes proxy-model information.
"""
from pathlib import Path
import os
os.environ.setdefault('HF_HOME', str(Path(__file__).resolve().parents[2] / '.hf-cache'))
import re
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from common import ROOT,read_data,dump,manifest,ranked,environment
from pairs import matched_replacements


def gradient_scores(h, y, p, refs):
    residual = p - y
    mean_w = (residual[refs,None] * h[refs]).mean(axis=0)
    mean_b = residual[refs].mean()
    return residual * (h @ mean_w + mean_b)


def main():
    started=time.time()
    ds,meta=read_data()
    texts=list(ds['pool']['sentence']); y=np.asarray(ds['pool']['label'])
    masks={
        'negation':np.array([bool(re.search(r"\b(?:not|never|no)\b|n't",t.lower())) for t in texts]),
        'contrast':np.array([bool(re.search(r'\b(?:but|although|however|yet)\b',t.lower())) for t in texts])}
    rng=np.random.default_rng(17); refs={}; used=set()
    for skill,mask in masks.items():
        ids=[]
        for label in [0,1]:
            candidates=[int(i) for i in np.flatnonzero(mask & (y==label)) if i not in used]
            if len(candidates)<64: raise ValueError(f'Not enough {skill} reference candidates')
            ids.extend(rng.choice(candidates,64,replace=False).tolist())
        refs[skill]=ids; used.update(ids)
    for skill in list(masks):
        ids=matched_replacements(refs[skill],texts,y,used,17)
        refs[skill+'_random']=ids;used.update(ids)
    proxy_ids=rng.choice([i for i in range(len(y)) if i not in used],4096,replace=False)
    used.update(proxy_ids.tolist())
    eligible=np.asarray([i for i in range(len(y)) if i not in used])
    dump(ROOT/'data/skill_roles.json',dict(references=refs,proxy_train=proxy_ids.tolist(),eligible=eligible.tolist()))
    pd.DataFrame([dict(role=k,index=i,text=texts[i],label=int(y[i]),review='') for k,ids in refs.items() for i in ids]).to_csv(ROOT/'reports/skill_references.csv',index=False)
    embfile=ROOT/'data/minilm_embeddings.npy'
    if not embfile.exists() or os.environ.get('RECOMPUTE_EMBEDDINGS') == '1':
        from huggingface_hub import HfApi,snapshot_download
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4)
        model_id='sentence-transformers/all-MiniLM-L6-v2'
        previous=ROOT/'data/embedding_provenance.json'
        revision=json_read(previous)['revision'] if previous.exists() else HfApi().model_info(model_id).sha
        model_path=snapshot_download(model_id,revision=revision,allow_patterns=['*.json','*.txt','*.safetensors','1_Pooling/*'],local_files_only=previous.exists())
        encoder=SentenceTransformer(model_path,device='cpu',local_files_only=True)
        encode_started=time.time()
        h=encoder.encode(texts,batch_size=128,show_progress_bar=True,normalize_embeddings=True,convert_to_numpy=True)
        np.save(embfile,h)
        dump(ROOT/'data/embedding_provenance.json',dict(model=model_id,revision=revision,pool_sha256=meta['splits']['pool']['sha256'],shape=list(h.shape),device='cpu',embedding_cpu_seconds=time.time()-encode_started))
    else:
        h=np.load(embfile)
        assert json_read(ROOT/'data/embedding_provenance.json')['pool_sha256']==meta['splits']['pool']['sha256']
    head=LogisticRegression(C=1,max_iter=1000,solver='lbfgs',random_state=17)
    head.fit(h[proxy_ids],y[proxy_ids])
    p=head.predict_proba(h)[:,1]
    np.savez_compressed(ROOT/'data/proxy_head.npz',coef=head.coef_,intercept=head.intercept_,probabilities=p)
    scores={}
    for skill,mask in masks.items():
        scores[skill+'_gradient']=gradient_scores(h,y,p,refs[skill])
        scores[skill+'_randomref_gradient']=gradient_scores(h,y,p,refs[skill+'_random'])
        scores[skill+'_embedding']=h @ h[refs[skill]].mean(axis=0)
        scores[skill+'_keyword']=mask.astype(float)
    np.savez_compressed(ROOT/'data/skill_scores.npz',**scores)
    summary=[]
    for seed in [17,42,101]:
        random_scores=np.random.default_rng(seed).random(len(y))
        for name,score in dict(skill_random=random_scores,**scores).items():
            ids=ranked(y,score,6534,seed,eligible)
            manifest(name,ids,seed,meta,dict(proxy='frozen MiniLM + logistic head',score=name,
                reference_rows=512,proxy_train_rows=4096,eligible_rows=len(eligible),
                reference_labels='original SST-2 labels; skill membership identified by lexical markers, not manually certified',
                gradient='raw dot product, including bias; no cosine normalization',
                total_cpu_seconds=time.time()-started))
            summary.append(dict(method=name,seed=seed,n=len(ids),negation_fraction=float(masks['negation'][ids].mean()),contrast_fraction=float(masks['contrast'][ids].mean()),positive_share=float(y[ids].mean())))
    pd.DataFrame(summary).to_csv(ROOT/'reports/skill_selection_summary.csv',index=False)
    dump(ROOT/'reports/skill_preparation.json',dict(status='complete',cpu_seconds=time.time()-started,
        gpu_used=False,embedding_cpu_seconds=json_read(ROOT/'data/embedding_provenance.json').get('embedding_cpu_seconds'),environment=environment(),proxy_train_accuracy=float(head.score(h[proxy_ids],y[proxy_ids])),
        reference_accuracy={k:float(head.score(h[v],y[v])) for k,v in refs.items()},
        caveat='Reference accuracy is descriptive, never used to select hyperparameters. No val/test used by selector.'))
    print('Skill preparation complete',flush=True)


def json_read(path):
    import json
    return json.loads(path.read_text())

if __name__=='__main__': main()
