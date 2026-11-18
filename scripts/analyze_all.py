"""Validate and summarize 84 completed runs without training or network access."""
import argparse,hashlib,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/enn-report-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/enn-report-cache')
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import accuracy_score,f1_score
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def control(part,name):
 if part=='part1':return 'random'
 if part=='part2':return 'pair_random' if name.startswith('pair_') else 'skill_random'
 return 'forget_matched_zero' if name.startswith('forget_') else 'skill_matched_random_v2'
def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--results',type=Path,default=ROOT/'results');a=p.parse_args()
 ds=load_from_disk(str(a.dataset));out=a.results/'analysis';out.mkdir(exist_ok=True)
 y=np.array(ds['test']['label']);texts=pd.Series(ds['test']['sentence'])
 masks={'negation':texts.str.contains(r"\b(?:not|never|no)\b|n't",case=False,regex=True).to_numpy(),'contrast':texts.str.contains(r'\b(?:but|although|however|yet)\b',case=False,regex=True).to_numpy()}
 rows=[];identities={};configs=[]
 for part,expected in [('part1',15),('part2',39),('part3',30)]:
  spec=json.loads((a.results/'inputs'/part/'snapshot.json').read_text())
  for split in ['pool','val','test']:
   assert digest(list(zip(ds[split]['orig_idx'],ds[split]['sentence'],ds[split]['label'])))==spec['splits'][split]['sha256']
  files=sorted((a.results/'raw'/part/'runs').glob('*/metrics.json'));assert len(files)==expected
  for f in files:
   d=json.loads(f.read_text());identity=json.loads((f.parent/'identity.json').read_text());configs.append(identity['config'])
   assert not d['pilot'] and d['split']=='test' and d['global_step']==615
   assert d['training_seed']==identity['seed'] and d['selection_seed']==d['training_seed']
   pred=pd.read_csv(f.parent/'predictions.csv');assert len(pred)==872 and np.array_equal(pred.label,y)
   assert pred.p_positive.between(0,1).all() and np.isfinite(pred.p_positive).all()
   assert np.array_equal((pred.p_positive>.5).astype(int),pred.prediction)
   m={'accuracy':accuracy_score(y,pred.prediction),'macro_f1':f1_score(y,pred.prediction,average='macro'),'positive_f1':f1_score(y,pred.prediction)}
   for k,v in m.items():assert abs(v-d['metrics'][k])<1e-12,(f,k)
   sel=json.loads((a.results/'selections'/part/f'{d["name"]}__s{d["selection_seed"]}.json').read_text());sha=sel.pop('sha256')
   assert digest(sel)==sha==d['selection_sha256']==identity['selection_sha256']
   ids=sel['indices'];assert len(set(ids))==len(ids)==6534 and min(ids)>=0 and max(ids)<len(ds['pool'])
   assert sel['pool_sha256']==spec['splits']['pool']['sha256']
   identities[part,d['name'],d['training_seed']]=set(ids)
   row=dict(part=part,name=d['name'],seed=d['training_seed'],run_id=f.parent.name,**m,seconds=d['last_session_seconds'])
   evals=[x['eval_accuracy'] for x in d['log_history'] if 'eval_accuracy' in x];row['best_val_accuracy']=max(evals)
   for k,mask in masks.items():row[k+'_n']=int(mask.sum());row[k+'_accuracy']=float(np.mean(pred.prediction[mask]==y[mask]));row[k+'_complement_accuracy']=float(np.mean(pred.prediction[~mask]==y[~mask]))
   rows.append(row)
 assert all(c==configs[0] for c in configs),'Training configs differ'
 runs=pd.DataFrame(rows);runs.to_csv(out/'all_runs.csv',index=False);summaries=[]
 for (part,name),g in runs.groupby(['part','name']):
  assert sorted(g.seed)==([202,303,404] if part=='part3' else [17,42,101])
  base=control(part,name);b=runs[(runs.part==part)&(runs.name==base)].set_index('seed');delta=(g.set_index('seed').accuracy-b.accuracy)*100
  ci=t.ppf(.975,2)*delta.std()/np.sqrt(3)
  s=dict(part=part,name=name,n=3,accuracy_pct=g.accuracy.mean()*100,sd_pp=g.accuracy.std()*100,macro_f1_pct=g.macro_f1.mean()*100,positive_f1_pct=g.positive_f1.mean()*100,baseline=base,delta_pp=delta.mean(),delta_t95_low=delta.mean()-ci,delta_t95_high=delta.mean()+ci,
    negation_pct=g.negation_accuracy.mean()*100,contrast_pct=g.contrast_accuracy.mean()*100,
    distinct_subsets=len({tuple(sorted(identities[part,name,int(seed)])) for seed in g.seed}))
  summaries.append(s)
 summary=pd.DataFrame(summaries);summary.to_csv(out/'all_summary.csv',index=False)
 comparisons=[]
 pairs=[('forget_dose05','forget_random'),('forget_dose10','forget_random'),('forget_all','forget_random'),('forget_dose05','forget_matched_zero'),('forget_dose10','forget_matched_zero'),('forget_all','forget_matched_zero'),('grad_nobias_mix25','grad_raw_mix25'),('grad_nobias_mix25','grad_randomref_nobias_mix25'),('grad_raw_mix25','skill_matched_random_v2'),('grad_nobias_mix25','skill_matched_random_v2'),('neg_embedding_mix25','skill_matched_random_v2')]
 for method,base in pairs:
  mg=runs[(runs.part=='part3')&(runs.name==method)].set_index('seed');bg=runs[(runs.part=='part3')&(runs.name==base)].set_index('seed')
  for seed in [202,303,404]:comparisons.append(dict(method=method,baseline=base,seed=seed,delta_pp=100*(mg.loc[seed,'accuracy']-bg.loc[seed,'accuracy']),shared_rows=len(identities['part3',method,seed]&identities['part3',base,seed])))
 pd.DataFrame(comparisons).to_csv(out/'followup_comparisons.csv',index=False)
 for title,mask in [('part1',summary.part=='part1'),('part2',summary.part=='part2'),('part3_forgetting',(summary.part=='part3')&summary.name.str.startswith('forget_')),('part3_skills',(summary.part=='part3')&~summary.name.str.startswith('forget_'))]:
  z=summary[mask].sort_values('accuracy_pct');fig,ax=plt.subplots(figsize=(11,max(4,len(z)*.4)))
  ax.errorbar(z.accuracy_pct,np.arange(len(z)),xerr=z.sd_pp,fmt='none',ecolor='#54616d',capsize=3);ax.scatter(z.accuracy_pct,np.arange(len(z)),s=50,c=['#267b70' if 'random' in n or 'matched_zero' in n else '#527aa3' for n in z.name]);ax.set_yticks(np.arange(len(z)),z.name);ax.grid(axis='x',alpha=.2);ax.set_xlim(65 if title=='part2' else 88,97);ax.set_xlabel('Test accuracy (%) ± SD, 3 seeds');ax.set_title(title);fig.tight_layout();fig.savefig(out/f'{title}_all_accuracy.png',dpi=160);plt.close(fig)
 (out/'verification.json').write_text(json.dumps(dict(runs=len(rows),conditions=len(summary),seeds_old=[17,42,101],seeds_new=[202,303,404],test_rows=len(y),test_slices={k:int(v.sum()) for k,v in masks.items()},metrics_recomputed=['accuracy','macro_f1','positive_f1'],config=configs[0],notes='SD is not a confidence interval. Paired t intervals are exploratory, unadjusted, fixed-test and n=3.'),indent=2))
 print(summary[summary.part=='part3'][['name','accuracy_pct','sd_pp','macro_f1_pct','negation_pct','contrast_pct']].to_string(index=False))
 print('Validated',len(rows),'runs; tables and figures:',out)
if __name__=='__main__':main()
