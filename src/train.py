"""Fine-tune ModernBERT on a subset of the SST-2 pool and log metrics.

Examples:
    python -m src.train --subset full --seed 0
    python -m src.train --subset random_s0 --seed 0
    python -m src.train --subset full --seed 0 --log-dynamics   # per-example training dynamics
"""
import argparse
import csv
import json
import math
import time
from datetime import datetime

import datasets
import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from src.config import MODEL_NAME, RESULTS_CSV, RUNS_DIR, TRAIN_CFG
from src.data import load_splits, load_subset

transformers.logging.set_verbosity_error()
(getattr(datasets, "disable_progress_bars", None) or datasets.disable_progress_bar)()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True, help="'full' or name of results/subsets/<name>.npy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="free-form label stored in results.csv")
    p.add_argument("--val-indices", default=None, help=".npy indices of validation used for epoch selection")
    p.add_argument("--subset-weights", default=None, help=".npy weights aligned with subset indices")
    p.add_argument("--log-dynamics", action="store_true",
                   help="save per-example gold-label probs on the train subset after each epoch")
    for k, v in TRAIN_CFG.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    return p.parse_args()


def tokenize(ds, tok, max_length):
    ds = ds.map(lambda b: tok(b["sentence"], truncation=True, max_length=max_length), batched=True)
    keep = ["input_ids", "attention_mask", "label"]
    return ds.remove_columns([c for c in ds.column_names if c not in keep]).rename_column("label", "labels")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        logits.append(out.logits.float().cpu())
        labels.append(batch["labels"].cpu())
    return torch.cat(logits), torch.cat(labels)


def metrics(logits, labels):
    preds = logits.argmax(-1).numpy()
    return dict(acc=accuracy_score(labels, preds), f1=f1_score(labels, preds))


def make_model_and_optim(n_train, args, device):
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, attn_implementation="sdpa"
    ).to(device)

    no_decay = ("bias", "LayerNorm.weight", "norm.weight")
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=args.lr)
    total_steps = args.epochs * math.ceil(n_train / args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * total_steps), total_steps)
    return model, opt, sched


def train_epoch(model, loader, opt, sched, device, desc="train"):
    model.train()
    running = 0.0
    scaler = torch.amp.GradScaler("cuda", enabled=not torch.cuda.is_bf16_supported())
    for step, batch in enumerate(tqdm(loader, desc=desc, leave=False, dynamic_ncols=True), 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        weights = batch.pop("sample_weight", None)
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            if weights is None:
                loss = model(**batch).loss
            else:
                labels = batch.pop("labels")
                logits = model(**batch).logits
                per_example = torch.nn.functional.cross_entropy(logits.float(), labels, reduction="none")
                loss = (per_example * weights).mean()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        running += loss.item()
    return running / step


def append_result(row):
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = "cuda"

    splits = load_splits()
    pool = splits["pool"]
    if args.subset == "full":
        idx = np.arange(len(pool))
    else:
        idx = load_subset(args.subset)
    train_raw = pool.select(idx)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = DataCollatorWithPadding(tok)
    train_ds = tokenize(train_raw, tok, args.max_length)
    if args.subset_weights:
        weights = np.load(args.subset_weights)
        if weights.shape != (len(train_ds),) or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("invalid sample weights")
        train_ds = train_ds.add_column("sample_weight", weights.astype(np.float32))
    val_raw = splits["val"]
    if args.val_indices:
        vi = np.load(args.val_indices)
        if len(np.unique(vi)) != len(vi) or (vi < 0).any() or (vi >= len(val_raw)).any():
            raise ValueError("invalid validation indices")
        val_raw = val_raw.select(vi)
    val_ds = tokenize(val_raw, tok, args.max_length)
    test_ds = tokenize(splits["test"], tok, args.max_length)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    mk_eval = lambda d: DataLoader(d, batch_size=args.eval_batch_size, collate_fn=collate)
    val_loader, test_loader = mk_eval(val_ds), mk_eval(test_ds)
    train_eval_loader = mk_eval(train_ds) if args.log_dynamics else None

    model, opt, sched = make_model_and_optim(len(train_ds), args, device)

    best = dict(val_acc=-1.0)
    dynamics = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, sched, device, desc=f"epoch {epoch}/{args.epochs}")
        val_m = metrics(*predict(model, val_loader, device))
        test_m = metrics(*predict(model, test_loader, device))
        print(f"epoch {epoch}: train_loss {train_loss:.4f} | "
              f"val acc {val_m['acc']:.4f} f1 {val_m['f1']:.4f} | test acc {test_m['acc']:.4f} f1 {test_m['f1']:.4f}")
        if val_m["acc"] > best["val_acc"]:
            best = dict(val_acc=val_m["acc"], val_f1=val_m["f1"], test_acc=test_m["acc"],
                        test_f1=test_m["f1"], best_epoch=epoch)

        if args.log_dynamics:
            logits, labels = predict(model, train_eval_loader, device)
            probs = logits.softmax(-1)
            dynamics.append(dict(gold_prob=probs[torch.arange(len(labels)), labels].numpy(),
                                 logits=logits.numpy()))

    elapsed = time.time() - t0
    run_name = f"{args.subset}_seed{args.seed}{'_' + args.tag if args.tag else ''}"
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if args.log_dynamics:
        np.savez_compressed(
            RUNS_DIR / f"dynamics_{run_name}.npz",
            pool_idx=idx,
            labels=np.asarray(train_raw["label"]),
            gold_prob=np.stack([d["gold_prob"] for d in dynamics]),
            logits=np.stack([d["logits"] for d in dynamics]),
        )

    row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), subset=args.subset, tag=args.tag,
               seed=args.seed, n_train=len(train_ds), **{k: round(v, 5) if isinstance(v, float) else v
                                                        for k, v in best.items()},
               train_time_s=round(elapsed, 1), config=json.dumps({k: getattr(args, k) for k in TRAIN_CFG}))
    append_result(row)
    print(f"best (by val): {best} | {elapsed:.0f}s")


if __name__ == "__main__":
    main()
