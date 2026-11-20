"""Create three OOF TF-IDF proxy scores and unsupervised long-text features."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import numpy as np
import sklearn
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/'experiments/01_cpu_selection/code'))
from common import read_data
ROOT=REPO/'results/proxy_mixture'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def oof(texts,y,seed):
    start=time.time();gold=np.zeros(len(y),dtype=np.float32);acc=[]
    folds=StratifiedKFold(3,shuffle=True,random_state=seed)
    for fold,(train,held) in enumerate(folds.split(np.zeros(len(y)),y)):
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True)
        x=vec.fit_transform([texts[i] for i in train]);z=vec.transform([texts[i] for i in held])
        model=SGDClassifier(loss='log_loss',alpha=1e-5,random_state=seed+fold,average=True)
        rng=np.random.default_rng(seed+fold)
        for _ in range(8):
            order=rng.permutation(len(train));model.partial_fit(x[order],y[train][order],classes=np.array([0,1]))
        prob=model.predict_proba(z);gold[held]=prob[np.arange(len(held)),y[held]]
        acc.append(float((prob.argmax(1)==y[held]).mean()))
        print(f'seed {seed}, fold {fold}: OOF accuracy={acc[-1]:.4f}',flush=True)
    if not np.isfinite(gold).all() or (gold<0).any() or (gold>1).any():raise ValueError('Invalid gold probabilities')
    return gold,{'seed':seed,'fold_accuracy':acc,'seconds':time.time()-start}


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',nargs='+',type=int,default=[17,42,101]);args=p.parse_args()
    ds,meta=read_data();texts=ds['pool']['sentence'];y=np.asarray(ds['pool']['label']);length=np.asarray([len(s.split()) for s in texts])
    ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'proxy').mkdir(exist_ok=True);started=time.time();records=[]
    for seed in args.seeds:
        result,info=oof(texts,y,seed);path=ROOT/'proxy'/f'gold_probability_s{seed}.npz'
        np.savez_compressed(path,p_gold=result)
        info.update({'sha256':sha(path),'pool_sha256':meta['splits']['pool']['sha256']});records.append(info)
    print('Building unsupervised TF-IDF clusters...',flush=True);t=time.time()
    eligible=np.flatnonzero(length>=8)
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=50000,sublinear_tf=True,dtype=np.float32)
    x=vec.fit_transform([texts[i] for i in eligible])
    km=MiniBatchKMeans(n_clusters=64,random_state=1729,batch_size=2048,n_init=3,max_iter=100,init_size=4096)
    cluster=np.full(len(y),-1,dtype=np.int16);cluster[eligible]=km.fit_predict(x).astype(np.int16)
    # Rare-word score uses document frequency in the complete pool and the same token regex as sklearn.
    token_re=re.compile(r'(?u)\b\w\w+\b');tokens=[token_re.findall(t.lower()) for t in texts]
    df=Counter(token for row in tokens for token in set(row));n=len(texts)
    rare=np.asarray([np.mean([np.log((n+1)/(df[w]+1))+1 for w in row]) if row else 0.0 for row in tokens],dtype=np.float32)
    path=ROOT/'proxy'/'lexical_features.npz';np.savez_compressed(path,cluster=cluster,rare_idf=rare)
    info={'pool_sha256':meta['splits']['pool']['sha256'],'seeds':records,
          'proxy':'TF-IDF unigrams+bigrams, SGDClassifier log-loss; 3-fold OOF, 8 epochs',
          'cluster':'MiniBatchKMeans 64 on long-text TF-IDF, seed 1729',
          'long_count':int(len(eligible)),'clusters_present':int(len(np.unique(cluster[eligible]))),
          'cluster_largest_share':float(np.bincount(cluster[eligible],minlength=64).max()/len(eligible)),
          'lexical_seconds':time.time()-t,'lexical_sha256':sha(path),'sklearn':sklearn.__version__,
          'total_seconds':time.time()-started}
    (ROOT/'proxy'/'metadata.json').write_text(json.dumps(info,indent=2)+'\n');print(json.dumps(info,indent=2),flush=True)

if __name__=='__main__':main()
