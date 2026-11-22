"""Fine-tune ModernBERT on a subset of the SST-2 pool and log metrics.

Examples:
    python -m src.train --subset full --seed 0
    python -m src.train --subset random_s0 --seed 0
    python -m src.train --subset full --seed 0 --log-dynamics   # per-example training dynamics
    python -m src.train --subset roots_s0 --seed 0 --teacher llm --tag kd_llm   # knowledge distillation
    python -m src.train --subset roots_s0 --seed 0 --save-probs --tag teacher  # be a teacher: data/teacher_self_roots_s0.npy

Distillation loss: alpha * CE(gold label) + (1 - alpha) * T^2 * KL(teacher || student) at temperature T
(Hinton et al., 2015). Teachers are pool-level probabilities data/teacher_<name>.npy, shape (n_pool, 2);
<name> may contain {subset}, e.g. --teacher self_{subset}.
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

from src.config import DATA_DIR, MODEL_NAME, RESULTS_CSV, RUNS_DIR, TRAIN_CFG
from src.data import load_splits, load_subset

transformers.logging.set_verbosity_error()
datasets.disable_progress_bars()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True, help="'full' or name of results/subsets/<name>.npy")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="", help="free-form label stored in results.csv")
    p.add_argument("--log-dynamics", action="store_true",
                   help="save per-example gold-label probs on the train subset after each epoch")
    p.add_argument("--teacher", default="", help="distill from data/teacher_<name>.npy")
    p.add_argument("--alpha", type=float, default=0.5, help="weight of the gold-label loss; 0 = teacher only")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--save-probs", action="store_true",
                   help="save train-subset probs of the best epoch as data/teacher_self_<subset>.npy")
    for k, v in TRAIN_CFG.items():
        p.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    return p.parse_args()


def tokenize(ds, tok, max_length):
    ds = ds.map(lambda b: tok(b["sentence"], truncation=True, max_length=max_length), batched=True)
    keep = ["input_ids", "attention_mask", "label", "teacher_probs"]
    return ds.remove_columns([c for c in ds.column_names if c not in keep]).rename_column("label", "labels")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
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


def distill_loss(logits, labels, teacher_probs, alpha, temperature):
    logits = logits.float()
    t_log = torch.log(teacher_probs.float().clamp_min(1e-8)) / temperature
    kl = torch.nn.functional.kl_div(torch.log_softmax(logits / temperature, -1), torch.log_softmax(t_log, -1),
                                    log_target=True, reduction="batchmean") * temperature ** 2
    return alpha * torch.nn.functional.cross_entropy(logits, labels) + (1 - alpha) * kl


def train_epoch(model, loader, opt, sched, device, desc="train", alpha=0.5, temperature=2.0):
    model.train()
    running = 0.0
    for step, batch in enumerate(tqdm(loader, desc=desc, leave=False, dynamic_ncols=True), 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        teacher = batch.pop("teacher_probs", None)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if teacher is None:
                loss = model(**batch).loss
            else:
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
                loss = distill_loss(logits, batch["labels"], teacher, alpha, temperature)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
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
    if args.teacher:
        teacher = np.load(DATA_DIR / f"teacher_{args.teacher.format(subset=args.subset)}.npy")[idx]
        assert np.isfinite(teacher).all(), "teacher has no probabilities for some rows of this subset"
        train_raw = train_raw.add_column("teacher_probs", teacher.astype(np.float32).tolist())

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    collate = DataCollatorWithPadding(tok)
    train_ds = tokenize(train_raw, tok, args.max_length)
    val_ds = tokenize(splits["val"], tok, args.max_length)
    test_ds = tokenize(splits["test"], tok, args.max_length)

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=g)
    mk_eval = lambda d: DataLoader(d, batch_size=args.eval_batch_size, collate_fn=collate)
    val_loader, test_loader = mk_eval(val_ds), mk_eval(test_ds)
    train_eval_loader = mk_eval(train_ds) if args.log_dynamics or args.save_probs else None

    model, opt, sched = make_model_and_optim(len(train_ds), args, device)

    best = dict(val_acc=-1.0)
    dynamics = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, sched, device, desc=f"epoch {epoch}/{args.epochs}",
                                 alpha=args.alpha, temperature=args.temperature)
        val_m = metrics(*predict(model, val_loader, device))
        test_m = metrics(*predict(model, test_loader, device))
        print(f"epoch {epoch}: train_loss {train_loss:.4f} | "
              f"val acc {val_m['acc']:.4f} f1 {val_m['f1']:.4f} | test acc {test_m['acc']:.4f} f1 {test_m['f1']:.4f}")
        if val_m["acc"] > best["val_acc"]:
            best = dict(val_acc=val_m["acc"], val_f1=val_m["f1"], test_acc=test_m["acc"],
                        test_f1=test_m["f1"], best_epoch=epoch)
            if args.save_probs:
                best_train_probs = predict(model, train_eval_loader, device)[0].softmax(-1).numpy()

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
    if args.save_probs:
        full = np.full((len(pool), 2), np.nan, dtype=np.float32)
        full[idx] = best_train_probs
        np.save(DATA_DIR / f"teacher_self_{args.subset}.npy", full)

    config = {k: getattr(args, k) for k in TRAIN_CFG}
    if args.teacher:
        config.update(teacher=args.teacher, alpha=args.alpha, temperature=args.temperature)
    row = dict(timestamp=datetime.now().isoformat(timespec="seconds"), subset=args.subset, tag=args.tag,
               seed=args.seed, n_train=len(train_ds), **{k: round(v, 5) if isinstance(v, float) else v
                                                        for k, v in best.items()},
               train_time_s=round(elapsed, 1), config=json.dumps(config))
    append_result(row)
    print(f"best (by val): {best} | {elapsed:.0f}s")


if __name__ == "__main__":
    main()
