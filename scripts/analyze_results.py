"""Recompute all metrics and audits from real predictions and selection manifests."""
import argparse, hashlib, json, re, itertools, os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/enn-analysis-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/enn-analysis-cache')
import numpy as np
import pandas as pd
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import t, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

def masks(texts):
    return {k:np.array([bool(re.search(p,s.lower())) for s in texts]) for k,p in
      [('negation',r"\b(?:not|never|no)\b|n't"),('contrast',r'\b(?:but|although|however|yet)\b')]}

def baseline(part,name):
    return 'random' if part=='part1' else ('pair_random' if name.startswith('pair_') else 'skill_random')

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    a=p.parse_args();root=a.results;out=root/'analysis';out.mkdir(exist_ok=True)
    ds=load_from_disk(str(a.dataset));y=np.array(ds['pool']['label']);yt=np.array(ds['test']['label'])
    texts=list(ds['pool']['sentence']);tm=masks(list(ds['test']['sentence']));pm=masks(texts)
    lengths=np.array([len(s.split()) for s in texts]);rows=[];preds={};subsets={};audits=[]
    probs=np.load(root/'inputs/part2/proxy_head.npz')['probabilities'];loss=-np.log(np.clip(np.where(y==1,probs,1-probs),1e-12,1))
    for part in ['part1','part2']:
        spec=json.loads((root/'inputs'/part/'snapshot.json').read_text())
        for split in ['pool','val','test']:
            assert digest(list(zip(ds[split]['orig_idx'],ds[split]['sentence'],ds[split]['label'])))==spec['splits'][split]['sha256']
        for f in sorted((root/'raw'/part/'runs').glob('*/metrics.json')):
            d=json.loads(f.read_text());seed=d['training_seed'];name=d['name'];pred=pd.read_csv(f.parent/'predictions.csv')
            assert not d['pilot'] and d['split']=='test' and d['global_step']==615
            assert np.array_equal(pred.label,yt) and len(pred)==872
            assert np.isfinite(pred.p_positive).all() and pred.p_positive.between(0,1).all()
            acc=accuracy_score(yt,pred.prediction);f1=f1_score(yt,pred.prediction,average='macro')
            assert abs(acc-d['metrics']['accuracy'])<1e-12 and abs(f1-d['metrics']['macro_f1'])<1e-12
            selection=json.loads((root/'selections'/part/f'{name}__s{d["selection_seed"]}.json').read_text())
            sha=selection.pop('sha256');assert digest(selection)==sha==d['selection_sha256']
            ids=np.array(selection['indices']);assert len(set(ids))==len(ids)==6534
            subsets[part,name,seed]=set(ids.tolist());preds[part,name,seed]=pred.prediction.to_numpy()
            history=[x for x in d['log_history'] if 'eval_accuracy' in x];best=max(x['eval_accuracy'] for x in history)
            row=dict(part=part,name=name,seed=seed,accuracy=acc,macro_f1=f1,positive_f1=d['metrics']['positive_f1'],
                best_val_accuracy=best,seconds=d['last_session_seconds'],test_positive_prediction_share=pred.prediction.mean())
            for label in [0,1]:row[f'recall_{label}']=np.mean(pred.prediction[yt==label]==label)
            for k,mask in tm.items():
                row[k+'_n']=int(mask.sum());row[k+'_accuracy']=float(np.mean(pred.prediction[mask]==yt[mask]));row[k+'_complement_accuracy']=float(np.mean(pred.prediction[~mask]==yt[~mask]))
            rows.append(row)
            audit=dict(part=part,name=name,seed=seed,mean_words=lengths[ids].mean(),median_words=np.median(lengths[ids]),
               positive_share=y[ids].mean(),proxy_error=np.mean((probs[ids]>=.5)!=y[ids]),proxy_loss=loss[ids].mean(),
               unique_texts=len({texts[i].strip() for i in ids}))
            for k,mask in pm.items():
                audit[k+'_share']=mask[ids].mean()
                for label in [0,1]:audit[f'{k}_share_label{label}']=mask[ids[y[ids]==label]].mean()
            audits.append(audit)
    runs=pd.DataFrame(rows);assert len(runs)==54
    runs.to_csv(out/'runs.csv',index=False);audit=pd.DataFrame(audits);audit.to_csv(out/'selection_audit.csv',index=False)
    summary=[]
    for (part,name),g in runs.groupby(['part','name']):
        assert sorted(g.seed)==[17,42,101]
        base=baseline(part,name);b=runs[(runs.part==part)&(runs.name==base)].set_index('seed')
        delta=(g.set_index('seed').accuracy-b.accuracy)*100
        interval=t.ppf(.975,2)*delta.std(ddof=1)/np.sqrt(3)
        sets=[subsets[part,name,int(seed)] for seed in g.seed]
        summary.append(dict(part=part,name=name,n=3,accuracy_pct=g.accuracy.mean()*100,sd_pp=g.accuracy.std()*100,
            macro_f1_pct=g.macro_f1.mean()*100,positive_f1_pct=g.positive_f1.mean()*100,baseline=base,delta_pp=delta.mean(),
            delta_s17_pp=delta[17],delta_s42_pp=delta[42],delta_s101_pp=delta[101],
            delta_t95_low=delta.mean()-interval,delta_t95_high=delta.mean()+interval,
            negation_pct=g.negation_accuracy.mean()*100,contrast_pct=g.contrast_accuracy.mean()*100,
            best_val_pct=g.best_val_accuracy.mean()*100,mean_seconds=g.seconds.mean(),
            distinct_subsets=len({tuple(sorted(s)) for s in sets}),
            mean_seed_jaccard=np.mean([len(s&z)/len(s|z) for s,z in itertools.combinations(sets,2)])))
    s=pd.DataFrame(summary);s.to_csv(out/'summary.csv',index=False)
    # Direct causal comparison for the pair intervention, distinct from full-pool random.
    paired=[]
    for control in ['pair_random','pair_matched_random','pair_broken_pairs']:
        for seed in [17,42,101]:
            delta=np.mean(preds['part2','pair_pairs',seed]==yt)-np.mean(preds['part2',control,seed]==yt)
            first=subsets['part2','pair_pairs',seed];second=subsets['part2',control,seed]
            paired.append(dict(control=control,seed=seed,delta_pp=100*delta,shared_rows=len(first&second),replaced_rows=len(first-second)))
    pd.DataFrame(paired).to_csv(out/'pair_comparisons.csv',index=False)
    # Pool-only diagnostic: no test used to define new selectors.
    scorefile=np.load(root/'inputs/part2/skill_scores.npz');corr=[]
    roles=json.loads((root/'inputs/part2/skill_roles.json').read_text());eligible=np.array(roles['eligible'])
    for name in scorefile.files:
        if 'gradient' not in name:continue
        for label in [0,1]:
            ids=eligible[y[eligible]==label]
            corr.append(dict(name=name,label=label,spearman_loss=spearmanr(scorefile[name][ids],loss[ids]).statistic))
    pd.DataFrame(corr).to_csv(out/'gradient_loss_correlations.csv',index=False)
    geometry=[]
    residual=probs-y
    for role,refs in roles['references'].items():
        key=role.replace('_random','_randomref')+'_gradient'
        projection=scorefile[key]/residual
        ids=np.array(sorted(subsets['part2',key,17]))
        row=dict(name=key,reference_bias_gradient=float(residual[refs].mean()),
            projection_min=float(projection[eligible].min()),projection_max=float(projection[eligible].max()),
            positive_projection_share=float((projection[eligible]>0).mean()))
        for label in [0,1]:
            ii=ids[y[ids]==label];row[f'proxy_error_label{label}']=float(np.mean((probs[ii]>=.5)!=y[ii]))
        geometry.append(row)
    pd.DataFrame(geometry).to_csv(out/'gradient_geometry.csv',index=False)

    forgetting=[]
    for seed in [17,42,101]:
        score=np.load(root/f'inputs/part1/forgetting_s{seed}.npz');ids=list(subsets['part1','heldout_forgetting',seed]);v=score['forgetting']
        forgetting.append(dict(seed=seed,nonzero_pool=int((v>0).sum()),nonzero_selected=int((v[ids]>0).sum()),zero_selected=int((v[ids]==0).sum()),max_score=int(v.max())))
    pd.DataFrame(forgetting).to_csv(out/'forgetting_audit.csv',index=False)
    leak=[]
    for left,right in [('pool','val'),('pool','test'),('val','test')]:
        l={x.strip().lower() for x in ds[left]['sentence']}
        leak.append(dict(left=left,right=right,right_exact_text_overlap=sum(x.strip().lower() in l for x in ds[right]['sentence'])))
    checks=dict(verified_runs=54,pool=len(y),val=len(ds['val']),test=len(yt),test_slices={k:int(v.sum()) for k,v in tm.items()},
        overlap_normalization='strip + lowercase, exact text only; not source-sentence overlap',split_text_overlaps=leak,
        pool_mean_words=float(lengths.mean()),pool_proxy_error=float(np.mean((probs>=.5)!=y)),pool_proxy_loss=float(loss.mean()),
        eligible_proxy_error=float(np.mean((probs[eligible]>=.5)!=y[eligible])),eligible_proxy_loss=float(loss[eligible].mean()),
        val_test_method_spearman=float(spearmanr(s.best_val_pct,s.accuracy_pct).statistic))
    (out/'checks.json').write_text(json.dumps(checks,indent=2))
    for part in ['part1','part2']:
        z=s[s.part==part].sort_values('accuracy_pct');fig,ax=plt.subplots(figsize=(10,max(4,len(z)*.4)))
        ax.barh(z.name,z.accuracy_pct,xerr=z.sd_pp,color=['#3d8c85' if 'random' in n else '#597ea7' for n in z.name],capsize=3)
        ax.set_xlim(65 if part=='part2' else 85,97);ax.set_xlabel('Test accuracy (%) ± SD across 3 runs');fig.tight_layout();fig.savefig(out/f'{part}_accuracy.png',dpi=160);plt.close(fig)
    print(s[['part','name','accuracy_pct','sd_pp','delta_pp','negation_pct','contrast_pct','distinct_subsets']].to_string(index=False))
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
