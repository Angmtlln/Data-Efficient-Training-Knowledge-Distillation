"""CPU-only, predefined followups. Gold labels; no test-driven selection."""
from pathlib import Path
import json,sys,shutil,hashlib
import numpy as np,pandas as pd
from datasets import load_from_disk
ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'02_hypotheses'
sys.path.insert(0,str(OLD/'code'))
from common import digest,quotas
from pairs import matched_replacements
N=6534;SEEDS=[202,303,404]
def save(name,ids,seed,provenance):
 ids=sorted(map(int,ids));assert len(ids)==len(set(ids))==N
 spec=json.loads((OLD/'data/snapshot.json').read_text())
 value=dict(name=name,selection_seed=seed,indices=ids,n=N,ratio=.1,pool_sha256=spec['splits']['pool']['sha256'],provenance=provenance)
 value['sha256']=digest(value);(ROOT/'selections'/f'{name}__s{seed}.json').write_text(json.dumps(value,indent=2))
def main():
 (ROOT/'selections').mkdir(exist_ok=True);(ROOT/'reports').mkdir(exist_ok=True)
 ds=load_from_disk(str(OLD/'data/sst2'));texts=list(ds['pool']['sentence']);y=np.array(ds['pool']['label']);lens=np.array([len(s.split()) for s in texts]);lengthbin=np.minimum(lens//4,8)
 spec=json.loads((OLD/'data/snapshot.json').read_text())
 assert digest(list(zip(ds['pool']['orig_idx'],texts,y.tolist())))==spec['splits']['pool']['sha256']
 quota=quotas(y,N);audit=[];groups={'forgetting':['forget_random','forget_matched_zero','forget_dose05','forget_dose10','forget_all'],
 'skills':['skill_matched_random_v2','grad_raw_mix25','grad_nobias_mix25','grad_randomref_nobias_mix25','neg_embedding_mix25']}
 # Fixed scoring proxy from prior s17, explicitly separated from new selection/training seeds.
 forget=np.load(ROOT.parent/'01_cpu_selection/data/forgetting_s17.npz')['forgetting'];signal=np.flatnonzero(forget>0)
 assert len(signal)==932
 roles=json.loads((OLD/'data/skill_roles.json').read_text());eligible=np.array(roles['eligible']);p=np.load(OLD/'data/proxy_head.npz')['probabilities'];loss=-np.log(np.clip(np.where(y==1,p,1-p),1e-12,1));correct=(p>=.5)==y
 variants=np.load(ROOT/'gradient_variants.npz');oldscore=np.load(OLD/'data/skill_scores.npz');scores={'grad_raw_mix25':oldscore['negation_gradient'],'grad_nobias_mix25':variants['negation_center0_bias0'],'grad_randomref_nobias_mix25':variants['negation_random_center0_bias0'],'neg_embedding_mix25':oldscore['negation_embedding']}
 # Proxy correctness is a separate stratum so every arm has identical error counts by class.
 difficulty=np.zeros(len(y),int)
 for label in [0,1]:
  for corr in [False,True]:
   ids=eligible[(y[eligible]==label)&(correct[eligible]==corr)]
   bounds=np.quantile(loss[ids],[.2,.4,.6,.8]);difficulty[ids]=np.searchsorted(bounds,loss[ids])
 strata={}
 for i in eligible:strata.setdefault((int(y[i]),int(lengthbin[i]),int(correct[i]),int(difficulty[i])),[]).append(int(i))
 sq={}
 for label in [0,1]:
  keys=sorted(k for k in strata if k[0]==label);counts=np.array([len(strata[k]) for k in keys]);raw=counts/counts.sum()*quota[label];q=np.floor(raw).astype(int)
  for idx in np.argsort(-(raw-q),kind='stable')[:quota[label]-q.sum()]:q[idx]+=1
  sq.update(zip(keys,q.tolist()))
 for seed in SEEDS:
  rng=np.random.default_rng(seed)
  target=signal[np.lexsort((rng.random(len(signal)),-forget[signal]))].tolist()
  controls=matched_replacements(target,texts,y,set(target),seed)
  excluded=set(target)|set(controls);filler=[]
  for label,n in quota.items():
   count=n-sum(y[i]==label for i in target)
   candidates=[i for i in range(len(y)) if y[i]==label and i not in excluded]
   filler.extend(rng.choice(candidates,count,replace=False).tolist())
  random=[]
  for label,n in quota.items():random.extend(rng.choice(np.flatnonzero(y==label),n,replace=False).tolist())
  selections={'forget_random':random}
  for name,m in [('forget_matched_zero',0),('forget_dose05',327),('forget_dose10',653),('forget_all',len(target))]:
   selections[name]=filler+target[:m]+controls[m:]
  for name,ids in selections.items():
   save(name,ids,seed,dict(design='fixed filler + paired class/length matched replacements',scoring_seed=17,nonzero_pool=len(signal),nonzero_selected=int((forget[ids]>0).sum()),note='random is full-pool control; matched_zero excludes all nonzero score rows; fixed OOF scores, new selection+training seeds'))
  bg=[];skillsets={name:[] for name in groups['skills']}
  for key in sorted(strata):
   count=sq[key]
   if count==0:continue
   ids=np.array(strata[key]);ids=ids[rng.permutation(len(ids))];m=int(round(.25*count));background=ids[:count-m];available=ids[count-m:]
   bg.extend(background.tolist());skillsets['skill_matched_random_v2'].extend(ids[:count].tolist())
   for name,score in scores.items():
    order=np.lexsort((rng.random(len(available)),-score[available]));chosen=available[order[:m]]
    skillsets[name].extend(background.tolist()+chosen.tolist())
  for name,ids in skillsets.items():
   assert [sum(y[i]==c for i in ids) for c in [0,1]]==[quota[c] for c in [0,1]]
   # All groups share the exact stratum histogram and at least ~75% rows.
   hist={k:0 for k in strata}
   for i in ids:hist[(int(y[i]),int(lengthbin[i]),int(correct[i]),int(difficulty[i]))]+=1
   assert hist==sq
   save(name,ids,seed,dict(design='25% targeted replacements; common background; label x lengthbin x proxy correctness x within-label/correctness loss quintile quotas',fixed_proxy_seed=17,common_background=len(bg),targeted_slots=N-len(bg),reference_role='negation_random' if 'randomref' in name else 'negation',note='raw gradient mixing+quotas is a required ablation; no test used'))
  selections.update(skillsets)
  for name,ids in selections.items():
   a=np.array(ids);row=dict(name=name,seed=seed,n=len(a),mean_words=lens[a].mean(),positive_share=y[a].mean(),proxy_error=np.mean(~correct[a]),nonzero_forgetting=int((forget[a]>0).sum()))
   for label in [0,1]:row[f'proxy_error_label{label}']=np.mean(~correct[a[y[a]==label]])
   audit.append(row)
 pd.DataFrame(audit).to_csv(ROOT/'reports/selection_audit.csv',index=False)
 (ROOT/'experiment_groups.json').write_text(json.dumps(groups,indent=2))
 print(pd.DataFrame(audit).to_string(index=False))
if __name__=='__main__':main()
