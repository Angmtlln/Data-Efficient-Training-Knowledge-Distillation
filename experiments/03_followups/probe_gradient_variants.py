from pathlib import Path
import json,numpy as np
from datasets import load_from_disk
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parent
p=ROOT.parent/'02_hypotheses';h=np.load(p/'data/minilm_embeddings.npy');phead=np.load(p/'data/proxy_head.npz')['probabilities'];roles=json.loads((p/'data/skill_roles.json').read_text());y=np.array(load_from_disk(str(p/'data/sst2'))['pool']['label']);res=phead-y;train=np.array(roles['proxy_train']);eligible=np.array(roles['eligible']);loss=-np.log(np.clip(np.where(y==1,phead,1-phead),1e-12,1))
centers=np.stack([(res[train[y[train]==c],None]*h[train[y[train]==c]]).mean(0) for c in [0,1]])
bs=np.array([res[train[y[train]==c]].mean() for c in [0,1]])
out={};rows=[]
for role in ['negation','negation_random']:
 refs=np.array(roles['references'][role]);w=(res[refs,None]*h[refs]).mean(0);b=res[refs].mean()
 for center in [False,True]:
  ww=w-centers[y[refs]].mean(0) if center else w
  bb=b-bs[y[refs]].mean() if center else b
  for bias in [False,True]:
   score=res*(h@ww+(bb if bias else 0))
   key=f'{role}_center{int(center)}_bias{int(bias)}';out[key]=score
   row=dict(name=key)
   for label in [0,1]:
    ids=eligible[y[eligible]==label];chosen=ids[np.argsort(score[ids])[-3000:]]
    row[f'loss_rho_{label}']=float(spearmanr(loss[ids],score[ids]).statistic);row[f'error_{label}']=float(np.mean((phead[chosen]>=.5)!=y[chosen]))
   rows.append(row)
np.savez_compressed(ROOT/'gradient_variants.npz',**out)
print(json.dumps(rows,indent=2))
(ROOT/'gradient_diagnostics.json').write_text(json.dumps(rows,indent=2))
