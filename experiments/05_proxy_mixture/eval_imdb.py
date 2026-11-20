"""Evaluate completed best checkpoints on fixed IMDb stress test; never fit or tune on it."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from datasets import load_from_disk,load_dataset
from sklearn.metrics import accuracy_score,f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];ROOT=REPO/'results/proxy_mixture'
sys.path.insert(0,str(HERE));from prepare_imdb import clean,REV
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'));from common import digest
NEW=('mix_random','mix_hard','mix_uncertain','mix_mid','mix_easy','mix_rare','mix_diverse')
BASE=('random','len8');SEEDS=(17,42,101)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--check-only',action='store_true');p.add_argument('--cpu',action='store_true');args=p.parse_args()
    manifest=json.loads((ROOT/'imdb_manifest.json').read_text())
    if manifest['revision']!=REV or len(manifest['indices'])!=2000 or len(set(manifest['indices']))!=2000:raise ValueError('Invalid external manifest')
    ds=load_from_disk(str(args.source)) if args.source else load_dataset('stanfordnlp/imdb',revision=REV,split='test')
    ids=manifest['indices'];truth=[int(ds[i]['label']) for i in ids];texts=[clean(ds[i]['text']) for i in ids]
    if digest(list(zip(ids,truth,texts)))!=manifest['data_sha256']:raise ValueError('IMDb source or sample differs from manifest')
    runs=[]
    for name in NEW+BASE:
        parent=ROOT/'runs' if name in NEW else REPO/'results/len_forgetting/runs'
        for seed in SEEDS:
            runid=f'{name}__ss{seed}__ts{seed}';p=parent/runid/'metrics.json'
            if not p.exists():raise ValueError(f'Missing completed run: {p}')
            info=json.loads(p.read_text());ckpt=Path(info['best_checkpoint'])
            if info['pilot'] or info['split']!='test' or not (ckpt/'model.safetensors').is_file():raise ValueError(f'Invalid run/checkpoint: {runid}')
            runs.append((runid,info,ckpt))
    print('Validated frozen IMDb sample and',len(runs),'completed model checkpoints',flush=True)
    if args.check_only:return
    gpu='cuda' if torch.cuda.is_available() and not args.cpu else ('mps' if torch.backends.mps.is_available() and not args.cpu else 'cpu')
    if gpu=='cpu' and not args.cpu:raise RuntimeError('GPU unavailable; --cpu required')
    tok=AutoTokenizer.from_pretrained(str(REPO/'experiments/01_cpu_selection/data/tokenizer'),local_files_only=True)
    encoded=tok(texts,truncation=True,max_length=64)
    features=[{k:v[i] for k,v in encoded.items()} for i in range(len(ids))]
    loader=DataLoader(features,batch_size=32,shuffle=False,collate_fn=DataCollatorWithPadding(tok))
    out=ROOT/'imdb_runs';out.mkdir(parents=True,exist_ok=True)
    for runid,info,ckpt in runs:
        result_dir=out/runid;result_dir.mkdir(exist_ok=True)
        if (result_dir/'metrics.json').exists():
            existing=json.loads((result_dir/'metrics.json').read_text())
            if existing['imdb_data_sha256']!=manifest['data_sha256'] or existing['selection_sha256']!=info['selection_sha256']:raise ValueError('Existing external results mismatch')
            print('Already complete:',runid,flush=True);continue
        start=time.time();model=AutoModelForSequenceClassification.from_pretrained(str(ckpt),attn_implementation='sdpa').to(gpu).eval();parts=[]
        with torch.inference_mode():
            for batch in loader:
                logits=model(**{k:v.to(gpu) for k,v in batch.items()}).logits
                parts.append(torch.softmax(logits.float(),dim=-1).cpu().numpy())
        prob=np.concatenate(parts);pred=prob.argmax(1);y=np.asarray(truth)
        with (result_dir/'predictions.csv').open('w',newline='') as f:
            w=csv.writer(f,lineterminator='\n');w.writerow(['imdb_idx','label','prediction','p_positive'])
            w.writerows(zip(ids,truth,pred.tolist(),prob[:,1].tolist()))
        metrics={'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro')),
                 'positive_f1':float(f1_score(y,pred))}
        result={'name':info['name'],'selection_seed':info['selection_seed'],'training_seed':info['training_seed'],
            'selection_sha256':info['selection_sha256'],'imdb_data_sha256':manifest['data_sha256'],
            'n':len(ids),'max_length':64,'checkpoint':str(ckpt),'metrics':metrics,'seconds':time.time()-start,'device':gpu}
        (result_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        print('Completed IMDb:',runid,'accuracy',metrics['accuracy'],flush=True)
        del model
        if gpu=='mps':torch.mps.empty_cache()
        if gpu=='cuda':torch.cuda.empty_cache()

if __name__=='__main__':main()
