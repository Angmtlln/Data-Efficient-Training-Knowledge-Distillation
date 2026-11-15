"""Cheap difficulty scores: train 1 epoch on a random 10% subset, then score the whole pool.

Expensive reference (scores.py): 3 epochs on all 65k examples + 3 passes over the pool.
Cheap variant: 1 epoch on 6.5k examples + 1 pass over the pool (~15x less compute).

    python -m src.analysis.cheap_scores
Output: data/cheap_scores.parquet (pool_idx, el2n_cheap, gold_prob_cheap)
"""
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
from src.train import make_model_and_optim, predict, tokenize, train_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="random_s0", help="subset the cheap scorer is trained on")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()
    args = SimpleNamespace(**{**TRAIN_CFG, "epochs": cli.epochs})
    set_seed(cli.seed)

    pool = load_splits()["pool"]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = DataCollatorWithPadding(tok)
    pool_ds = tokenize(pool, tok, args.max_length)
    train_idx = load_subset(cli.subset)

    t0 = time.time()
    g = torch.Generator().manual_seed(cli.seed)
    train_loader = DataLoader(pool_ds.select(train_idx), batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate, generator=g)
    model, opt, sched = make_model_and_optim(len(train_idx), args, "cuda")
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, sched, "cuda", desc=f"cheap scorer epoch {epoch}")
        print(f"epoch {epoch}: train_loss {loss:.4f}")

    logits, labels = predict(model, DataLoader(pool_ds, batch_size=args.eval_batch_size, collate_fn=collate), "cuda")
    probs = logits.softmax(-1).numpy()
    labels = labels.numpy()
    el2n = np.linalg.norm(probs - np.eye(2)[labels], axis=1)
    print(f"cheap scoring took {time.time() - t0:.0f}s, pool acc of cheap scorer {(probs.argmax(1) == labels).mean():.4f}")

    pd.DataFrame(dict(pool_idx=pool["pool_idx"], el2n_cheap=el2n,
                      gold_prob_cheap=probs[np.arange(len(labels)), labels])).to_parquet(DATA_DIR / "cheap_scores.parquet")

    ref = pd.read_parquet(DATA_DIR / "scores.parquet")
    rho = pd.Series(el2n).corr(ref.el2n, method="spearman")
    k = int(0.1 * len(ref))
    top = lambda s: set(np.argsort(-np.asarray(s))[:k])
    print(f"Spearman(el2n_cheap, el2n_full) = {rho:.3f}; overlap of top-10% hardest = "
          f"{len(top(el2n) & top(ref.el2n)) / k:.3f}")


if __name__ == "__main__":
    main()
