"""A small SST-2 proxy and exact per-example linear-head gradients."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup, set_seed
from src.config import DATA_DIR
from src.data import load_splits, load_subset
from src.train import tokenize

MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'


def head_gradients(model, loader, device):
    """Gradient of CE loss w.r.t. both linear head weights and bias in eval mode."""
    model.eval(); result=[]; captured=[]
    hook = model.classifier.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach()))
    try:
        with torch.inference_mode():
            for batch in loader:
                batch = {k:v.to(device) for k,v in batch.items()}
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits.float()
                h = captured.pop().float(); p = logits.softmax(dim=-1)
                p[torch.arange(len(p), device=device), batch['labels']] -= 1
                h = torch.cat([h, torch.ones((len(h), 1), device=device)], dim=1)
                grad = (p[:,:,None] * h[:,None,:]).reshape(len(h), -1)
                result.append(grad.cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(result).astype(np.float32)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,default=0); p.add_argument('--eval-batch-size',type=int,default=128)
    args=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('A CUDA GPU is required')
    set_seed(args.seed); start=time.perf_counter(); splits=load_splits(); pool=splits['pool']; val=splits['val']
    if not np.array_equal(pool['pool_idx'],np.arange(len(pool))): raise ValueError('pool order mismatch')
    # Split only original val; separate data for gradient selection and epoch choice.
    from sklearn.model_selection import train_test_split
    select_idx, eval_idx = train_test_split(np.arange(len(val)),test_size=.5,random_state=42,stratify=val['label'])
    assert not set(select_idx).intersection(eval_idx) and len(select_idx)+len(eval_idx)==len(val)
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    np.save(DATA_DIR/'gradient_selection_val_idx.npy',select_idx); np.save(DATA_DIR/'gradient_epoch_val_idx.npy',eval_idx)
    tok=AutoTokenizer.from_pretrained(MODEL); collate=DataCollatorWithPadding(tok)
    pool_ds=tokenize(pool,tok,64); val_ds=tokenize(val.select(select_idx),tok,64)
    idx=load_subset('random_s0'); train=DataLoader(pool_ds.select(idx),batch_size=32,shuffle=True,collate_fn=collate,generator=torch.Generator().manual_seed(args.seed))
    model=AutoModelForSequenceClassification.from_pretrained(MODEL,num_labels=2).to('cuda')
    opt=torch.optim.AdamW(model.parameters(),lr=5e-5,weight_decay=.01)
    sched=get_linear_schedule_with_warmup(opt,int(.06*len(train)),len(train))
    scaler=torch.amp.GradScaler('cuda',enabled=not torch.cuda.is_bf16_supported())
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model.train()
    for batch in train:
        batch={k:v.to('cuda') for k,v in batch.items()}
        with torch.autocast('cuda',dtype=dtype): loss=model(**batch).loss
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True)
    train_seconds=time.perf_counter()-start; start=time.perf_counter()
    loader=lambda d: DataLoader(d,batch_size=args.eval_batch_size,collate_fn=collate)
    pool_grad=head_gradients(model,loader(pool_ds),'cuda'); val_grad=head_gradients(model,loader(val_ds),'cuda')
    np.save(DATA_DIR/'gradient_pool.npy',pool_grad); np.save(DATA_DIR/'gradient_val_mean.npy',val_grad.mean(axis=0))
    info={'proxy_model':MODEL,'proxy_seed':args.seed,'proxy_train_subset':'random_s0','proxy_n':len(idx),'pool_n':len(pool),'selection_val_n':len(select_idx),'epoch_val_n':len(eval_idx),'proxy_train_seconds':train_seconds,'scoring_seconds':time.perf_counter()-start}
    (DATA_DIR/'gradient_proxy_info.json').write_text(json.dumps(info,indent=2))
    print(info,flush=True)

if __name__ == '__main__': main()
