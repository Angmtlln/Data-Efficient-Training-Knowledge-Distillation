"""Fixed 32-example packs keep each selected pair together, without repeated rows."""
import numpy as np
import torch
from transformers import Trainer, DataCollatorWithPadding


def make_packs(indices,pairs,seed,paired,batch_size=32):
    rng=np.random.default_rng(seed);ids=set(indices)
    assert len(ids)==len(indices)
    used=set();units=[]
    if paired:
        for a,b in pairs:
            assert a in ids and b in ids and a!=b and not ({a,b}&used)
            used.update([a,b]);units.append([a,b])
    units.extend([[i] for i in indices if i not in used]);rng.shuffle(units)
    # Fill from units while preserving pairs; singleton rows fill odd spaces.
    packs=[];current=[]
    for unit in units:
        if len(current)+len(unit)>batch_size:
            # At most one empty slot: use one remaining singleton later.
            packs.append(current);current=[]
        current.extend(unit)
        if len(current)==batch_size:packs.append(current);current=[]
    if current:packs.append(current)
    # Repack deterministically with pair units first into shuffled slot pairs.
    # This guarantees ceil(N / batch_size) steps and no underfilled middle batches.
    if paired:
        pair_units=[u for u in units if len(u)==2];singles=[u[0] for u in units if len(u)==1]
        packs=[[] for _ in range((len(indices)+batch_size-1)//batch_size)]
        slots=[i for i in range(len(indices)//batch_size) for _ in range(batch_size//2)]
        remainder=len(indices)%batch_size
        if remainder:slots += [len(packs)-1]*(remainder//2)
        rng.shuffle(slots)
        for u,slot in zip(pair_units,slots):packs[slot].extend(u)
        assert len(pair_units)<=len(slots)
        k=0
        for j,pack in enumerate(packs):
            capacity=min(batch_size,len(indices)-j*batch_size);need=capacity-len(pack)
            pack.extend(singles[k:k+need]);k+=need;rng.shuffle(pack)
        assert k==len(singles)
    flat=[i for p in packs for i in p]
    assert len(flat)==len(indices) and set(flat)==ids
    assert len(packs)==(len(indices)+batch_size-1)//batch_size
    return packs

class PackedDataset(torch.utils.data.Dataset):
    def __init__(self,dataset,indices,pairs,seed,paired):
        self.dataset=dataset;self.packs=make_packs(indices,pairs,seed,paired)
        self.pair_id={i:n for n,p in enumerate(pairs) for i in p} if paired else {}
    def __len__(self):return len(self.packs)
    def __getitem__(self,index):
        return {'packed':[dict(self.dataset[i],pair_id=self.pair_id.get(i,-1)) for i in self.packs[index]]}

class PairCollator:
    def __init__(self,tokenizer):self.padding=DataCollatorWithPadding(tokenizer)
    def __call__(self,features):
        if 'packed' in features[0]:
            assert len(features)==1,'One pack contains 32 real examples'
            features=features[0]['packed']
        groups={};clean=[]
        for i,f in enumerate(features):
            label=int(f.get('labels',f.get('label')))
            if f.get('pair_id',-1)>=0:groups.setdefault(f['pair_id'],[]).append((i,label))
            row={k:f[k] for k in ['input_ids','attention_mask','token_type_ids'] if k in f}
            row['labels']=label;clean.append(row)
        pairs=[]
        for g in groups.values():
            assert len(g)==2 and {y for _,y in g}=={0,1}
            pairs.append([next(i for i,y in g if y==1),next(i for i,y in g if y==0)])
        batch=self.padding(clean);batch['pair_indices']=torch.tensor(pairs,dtype=torch.long).reshape(-1,2)
        return batch

def pair_penalty(logits,pairs,margin):
    if pairs.numel()==0:return logits.sum()*0
    p=logits.softmax(-1)[:,1]
    return torch.relu(margin-p[pairs[:,0]]+p[pairs[:,1]]).mean()

class PairTrainer(Trainer):
    def __init__(self,*args,rank_weight=0.,rank_margin=.2,**kwargs):
        super().__init__(*args,**kwargs);self.rank_weight=rank_weight;self.rank_margin=rank_margin
        self.model_accepts_loss_kwargs=False
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        pairs=inputs.pop('pair_indices');outputs=model(**inputs)
        rank=pair_penalty(outputs.logits,pairs,self.rank_margin)
        loss=outputs.loss+self.rank_weight*rank
        if model.training and self.state.global_step%20==0:
            self.log({'train_ce':outputs.loss.detach().float().item(),'train_pair_penalty':rank.detach().float().item(),'pairs_in_batch':len(pairs)})
        return (loss,outputs) if return_outputs else loss
