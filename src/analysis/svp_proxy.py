"""Train a one-epoch proxy and save pool probabilities and final-layer CLS vectors."""
import argparse
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed

from src.config import DATA_DIR, MODEL_NAME, TRAIN_CFG
from src.data import load_splits, load_subset
from src.train import make_model_and_optim, tokenize, train_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--subset', default='random_s0')
    parser.add_argument('--eval-batch-size', type=int, default=64)
    cli = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Proxy training requires a CUDA GPU')
    set_seed(cli.seed)
    args = SimpleNamespace(**{**TRAIN_CFG, 'epochs': 1})
    pool = load_splits()['pool']
    if list(pool['pool_idx']) != list(range(len(pool))):
        raise ValueError('Unexpected pool order')
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = DataCollatorWithPadding(tok)
    ds = tokenize(pool, tok, args.max_length)
    idx = load_subset(cli.subset)
    loader = DataLoader(ds.select(idx), batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(cli.seed))
    model, opt, sched = make_model_and_optim(len(idx), args, 'cuda')
    start = time.perf_counter()
    loss = train_epoch(model, loader, opt, sched, 'cuda', desc='proxy')
    print(f'proxy train loss {loss:.4f}; {time.perf_counter()-start:.1f}s')
    model.eval()
    probabilities, embeddings = [], []
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=cli.eval_batch_size, collate_fn=collate):
            labels = batch.pop('labels').numpy()
            batch = {key: value.to('cuda') for key, value in batch.items()}
            with torch.autocast('cuda', dtype=dtype):
                out = model(**batch, output_hidden_states=True)
            probs = out.logits.float().softmax(-1).cpu().numpy()
            probabilities.append(probs[np.arange(len(labels)), labels])
            embeddings.append(out.hidden_states[-1][:, 0, :].float().cpu().numpy())
    pd.DataFrame({'pool_idx': pool['pool_idx'], 'gold_prob_cheap': np.concatenate(probabilities)}).to_parquet(DATA_DIR / 'svp_proxy_scores.parquet', index=False)
    np.save(DATA_DIR / 'svp_embeddings.npy', np.concatenate(embeddings))
    print(f'proxy scoring and training: {time.perf_counter()-start:.1f}s')


if __name__ == '__main__':
    main()
