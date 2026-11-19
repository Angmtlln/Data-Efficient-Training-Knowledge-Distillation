"""Frozen small sentence encoder and a trainable small proxy for forgetting events."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, DataCollatorWithPadding, get_linear_schedule_with_warmup, set_seed
from src.data import load_splits
from src.train import tokenize
from src.config import DATA_DIR

MODEL = 'sentence-transformers/paraphrase-MiniLM-L3-v2'


def embed(pool, device, batch_size):
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(pool), batch_size):
            batch = tok(pool[start:start+batch_size]['sentence'], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
            h = model(**batch).last_hidden_state
            mask = batch['attention_mask'].unsqueeze(-1)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            parts.append(pooled.float().cpu().numpy())
            if start % (batch_size * 100) == 0: print(f'embeddings: {start}/{len(pool)}', flush=True)
    x = np.concatenate(parts)
    np.save(DATA_DIR / 'svp_minilm_l3.npy', x)
    print('saved', x.shape, flush=True)


def forgetting(pool, device, batch_size, epochs, seed):
    """Record correctness on each training presentation, before its optimizer update."""
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = tokenize(pool, tok, 128)
    base = base.add_column('pool_idx', list(range(len(base))))
    collate = DataCollatorWithPadding(tok)
    loader = DataLoader(base, batch_size=batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=.01)
    sched = get_linear_schedule_with_warmup(optimizer, int(.06*epochs*len(loader)), epochs*len(loader))
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    correct = np.zeros((epochs, len(pool)), dtype=bool)
    start = time.time()
    for epoch in range(epochs):
        model.train(); total=0.
        for batch in loader:
            ids = batch.pop('pool_idx').numpy()
            batch = {k:v.to(device) for k,v in batch.items()}
            with torch.autocast('cuda', dtype=dtype):
                out = model(**batch)
            correct[epoch, ids] = (out.logits.argmax(-1) == batch['labels']).detach().cpu().numpy()
            scaler.scale(out.loss).backward(); scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer); scaler.update(); sched.step(); optimizer.zero_grad(set_to_none=True)
            total += out.loss.item()
        print(f'epoch {epoch+1}/{epochs}, loss={total/len(loader):.4f}, correct={correct[epoch].mean():.4f}, elapsed={time.time()-start:.1f}s', flush=True)
    np.savez_compressed(DATA_DIR / 'svp_forgetting_proxy.npz', pool_idx=np.arange(len(pool)), gold_prob=correct.astype('float32'))
    print('never learned:', (~correct.any(axis=0)).sum(), 'forgotten:', (correct[:-1] & ~correct[1:]).sum(), flush=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('method', choices=['embed','forgetting'])
    p.add_argument('--batch-size', type=int, default=64); p.add_argument('--epochs', type=int, default=3); p.add_argument('--seed', type=int, default=0)
    args=p.parse_args(); pool=load_splits()['pool']
    if not np.array_equal(pool['pool_idx'], np.arange(len(pool))): raise ValueError('pool_idx order mismatch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.method == 'embed': embed(pool, device, args.batch_size)
    else:
        if device != 'cuda': raise RuntimeError('Training the proxy requires CUDA')
        forgetting(pool, device, args.batch_size, args.epochs, args.seed)

if __name__ == '__main__': main()
