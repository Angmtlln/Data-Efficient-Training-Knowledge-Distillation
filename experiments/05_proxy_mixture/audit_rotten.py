"""Audit a candidate external test set for overlap with SST-2 (no model predictions)."""
import argparse
import json
from pathlib import Path
import sys
from datasets import load_dataset,load_from_disk

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1];ROOT=REPO/'results/proxy_mixture'
sys.path.insert(0,str(HERE));from prepare_imdb import tokens
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'));from common import read_data,digest
REV='aa13bc287fa6fcab6daf52f0dfb9994269ffea28'


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);args=p.parse_args()
    ext=load_from_disk(str(args.source)) if args.source else load_dataset('cornell-movie-review-data/rotten_tomatoes',revision=REV,split='test')
    ds,meta=read_data();norm=lambda s:' '.join(tokens(s))
    pool=set(map(norm,ds['pool']['sentence']));all_data=set(pool)
    for split in ('val','test'):all_data.update(map(norm,ds[split]['sentence']))
    phrases=set(tuple(norm(s).split()) for s in all_data if len(norm(s).split())>=8)
    ext_norm=list(map(norm,ext['text']));contained=0
    for text in ext_norm:
        words=text.split();contained+=any(tuple(words[i:j]) in phrases for i in range(len(words)) for j in range(i+8,len(words)+1))
    result={'source':'cornell-movie-review-data/rotten_tomatoes','revision':REV,'split':'test','n':len(ext),
        'source_sha256':digest(list(zip(ext['text'],ext['label']))),
        'sst2_pool_sha256':meta['splits']['pool']['sha256'],'normalization':'lowercase ASCII alphanumeric tokens joined by space',
        'exact_pool':sum(t in pool for t in ext_norm),'exact_any_sst2_split':sum(t in all_data for t in ext_norm),
        'contains_entire_sst2_phrase_ge8_tokens':contained,
        'caveat':'Exact text and token containment are not a full audit of common source documents.'}
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'rotten_overlap.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
