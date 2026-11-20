"""Fix a stratified IMDb test sample and audit text overlap, without model outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import numpy as np
from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'))
from common import read_data,digest
ROOT=REPO/'results/proxy_mixture'
REV='e6281661ce1c48d982bc483cf8a173c1bbeb5d31'


def clean(s):return re.sub(r'<br\s*/?>',' ',s,flags=re.I).strip()
def tokens(s):return re.findall(r'[a-z0-9]+',s.lower())


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);args=p.parse_args()
    imdb=load_from_disk(str(args.source)) if args.source else load_dataset('stanfordnlp/imdb',revision=REV,split='test')
    ds,meta=read_data();labels=np.asarray(imdb['label']);rng=np.random.default_rng(20261121)
    ids=sorted(np.concatenate([rng.choice(np.flatnonzero(labels==label),1000,replace=False) for label in (0,1)]).tolist())
    texts=[clean(imdb[int(i)]['text']) for i in ids];truth=[int(labels[i]) for i in ids]
    norm=lambda s:' '.join(tokens(s))
    sources=[s for split in ('pool','val','test') for s in ds[split]['sentence']]
    normalized=set(map(norm,sources));exact=sum(norm(s) in normalized for s in texts)
    grams=set()
    for t in sources:
        w=tokens(t);grams.update(tuple(w[i:i+8]) for i in range(len(w)-7))
    overlap=0
    for t in texts:
        w=tokens(t);overlap+=any(tuple(w[i:i+8]) in grams for i in range(len(w)-7))
    tok=AutoTokenizer.from_pretrained(str(REPO/'experiments/01_cpu_selection/data/tokenizer'),local_files_only=True)
    lengths=[len(z) for z in tok(texts,add_special_tokens=True,truncation=False,verbose=False)['input_ids']]
    data=list(zip(ids,truth,texts));info={'source':'stanfordnlp/imdb','revision':REV,'split':'test','selection':'1000 per class without replacement, seed 20261121',
        'indices':ids,'n':len(ids),'label_counts':{'0':1000,'1':1000},'clean':'remove HTML br tags and trim',
        'data_sha256':digest(data),'sst2_pool_sha256':meta['splits']['pool']['sha256'],
        'normalized_exact_sst2_any_split':exact,'containing_any_8token_ngram_from_sst2':overlap,
        'overlap_caveat':'Common 8-grams are not proof of document overlap; no source identity or pretraining contamination audit.',
        'mean_tokens_untruncated':float(np.mean(lengths)),'median_tokens_untruncated':float(np.median(lengths)),
        'over_64_tokens':int(sum(x>64 for x in lengths)),'target_max_length':64}
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'imdb_manifest.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps({k:v for k,v in info.items() if k!='indices'},indent=2),flush=True)

if __name__=='__main__':main()
