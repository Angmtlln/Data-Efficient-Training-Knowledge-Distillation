"""Out-of-fold predicted probabilities on the whole pool (input for label-noise detection).

    python -m src.analysis.cv_predict --folds 3 --epochs 2
Output: data/oof_probs.npy  (n_pool, 2)
"""
import argparse
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, set_seed

from src.config import DATA_DIR, MODEL_NAME, TRAIN_CFG
from src.data import load_splits
from src.train import make_model_and_optim, predict, tokenize, train_epoch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    cli = p.parse_args()
    args = SimpleNamespace(**{**TRAIN_CFG, "epochs": cli.epochs})
    set_seed(cli.seed)

    pool = load_splits()["pool"]
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = DataCollatorWithPadding(tok)
    ds = tokenize(pool, tok, args.max_length)
    labels = np.asarray(pool["label"])

    oof = np.zeros((len(pool), 2), dtype=np.float32)
    skf = StratifiedKFold(cli.folds, shuffle=True, random_state=cli.seed)
    for k, (tr, ho) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        g = torch.Generator().manual_seed(cli.seed + k)
        tr_loader = DataLoader(ds.select(tr), batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate, generator=g)
        ho_loader = DataLoader(ds.select(ho), batch_size=args.eval_batch_size, collate_fn=collate)
        model, opt, sched = make_model_and_optim(len(tr), args, "cuda")
        for epoch in range(args.epochs):
            loss = train_epoch(model, tr_loader, opt, sched, "cuda")
            print(f"fold {k} epoch {epoch + 1}: train_loss {loss:.4f}", flush=True)
        logits, _ = predict(model, ho_loader, "cuda")
        oof[ho] = logits.softmax(-1).numpy()
        print(f"fold {k}: held-out acc {(oof[ho].argmax(1) == labels[ho]).mean():.4f}", flush=True)
        del model, opt

    np.save(DATA_DIR / "oof_probs.npy", oof)
    print(f"overall OOF acc {(oof.argmax(1) == labels).mean():.4f}")


if __name__ == "__main__":
    main()
