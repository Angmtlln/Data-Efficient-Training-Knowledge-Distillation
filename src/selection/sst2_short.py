"""Rebuild the long-phrase and root-contrast selections used in the short T4 runs."""
import argparse
import csv
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from sklearn.random_projection import GaussianRandomProjection
from threadpoolctl import threadpool_limits
from transformers import AutoModel, AutoTokenizer

from src.config import BUDGET_FRACTION, MODEL_NAME
from src.data import load_splits
from src.selection.root_contrasts import contrast_subset, tokens


def root_mask(texts, labels, source):
    if hashlib.sha256(source.read_bytes()).hexdigest() != "7a4b1cfdd65be1dc48339404db86528bb2427e1d8772860ef838b76b8c38c4a8":
        raise ValueError("Unexpected SetFit/sst2 train.jsonl")
    train = pd.read_json(source, lines=True)
    keys = {("".join(tokens(t)), label) for t, label in zip(train.text, train.label)}
    return np.array([("".join(tokens(t)), label) in keys for t, label in zip(texts, labels)])


def root_subsets(texts, labels, roots, budget, seed):
    eligible = np.flatnonzero(roots)
    npos = round(budget * labels[eligible].mean())
    rng = np.random.default_rng(seed)
    groups = [rng.choice(eligible[labels[eligible] == c], n, replace=False)
              for c, n in [(0, budget - npos), (1, npos)]]
    base = np.random.default_rng(seed).permutation(np.concatenate(groups))
    modified, swaps = contrast_subset(texts, labels, roots, base, seed)
    return {"roots": base, "roots_contrast": modified}, swaps


def pilot_features(texts, long_ids, words):
    encoder_name = "sentence-transformers/paraphrase-MiniLM-L3-v2"
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name).cpu().eval()
    vectors = []
    with torch.inference_mode():
        for batch in np.array_split(long_ids, int(np.ceil(len(long_ids) / 96))):
            inputs = tokenizer([texts[i] for i in batch], padding=True, truncation=True,
                               max_length=128, return_tensors="pt")
            mask = inputs.attention_mask[..., None]
            vectors.append(((encoder(**inputs).last_hidden_state * mask).sum(1) / mask.sum(1)).numpy())
    projection = GaussianRandomProjection(n_components=32, random_state=42)
    matrix = projection.fit_transform(normalize(np.concatenate(vectors)))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ratios = np.full(len(texts), np.nan)
    for batch in np.array_split(long_ids, 40):
        encoded = tokenizer([texts[i] for i in batch], add_special_tokens=False, truncation=False)
        ratios[batch] = np.array([len(row) for row in encoded.input_ids]) / words[batch]
    return matrix, ratios


def fragmentation(reference, long_ids, labels, words, ratios, seed):
    rng = np.random.default_rng(seed)
    selected = []
    strata, quotas = np.unique(np.column_stack([labels[reference], words[reference]]),
                               axis=0, return_counts=True)
    for (label, length), quota in zip(strata, quotas):
        group = long_ids[(labels[long_ids] == label) & (words[long_ids] == length)]
        lo, hi = np.quantile(ratios[group], [0.05, 0.95])
        bins = np.zeros(len(group), int) if lo == hi else np.digitize(
            ratios[group], [lo + (hi - lo) / 3, lo + 2 * (hi - lo) / 3])
        groups = sorted((group[bins == k] for k in np.unique(bins)), key=len)
        for j, members in enumerate(groups):
            take = min(len(members), quota // (len(groups) - j))
            selected.append(rng.choice(members, take, replace=False))
            quota -= take
    return np.concatenate(selected)


def cluster_groups(matrix, count, seed):
    model = MiniBatchKMeans(n_clusters=count, init="random", n_init=1, max_iter=35,
                            batch_size=max(4096, count), random_state=seed)
    with threadpool_limits(limits=2):
        groups = model.fit_predict(matrix)
    counts = np.bincount(groups, minlength=count)
    empty = list(np.flatnonzero(counts == 0))
    distance = np.sum((matrix - model.cluster_centers_[groups]) ** 2, axis=1)
    for i in np.argsort(-distance, kind="stable"):
        if not empty:
            break
        old = groups[i]
        if counts[old] > 1:
            counts[old] -= 1
            groups[i] = empty.pop()
    means = np.zeros((count, matrix.shape[1]), dtype=np.float64)
    np.add.at(means, groups, matrix)
    means /= np.bincount(groups, minlength=count)[:, None]
    distance = np.sum((matrix - means[groups]) ** 2, axis=1)
    order = np.argsort(groups, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(groups[order])) + 1), distance


def pilot_subsets(matrix, ratios, labels, words, roots, budget, seed):
    long_ids = np.flatnonzero(words >= 8)
    reference = np.random.default_rng(seed).choice(long_ids, budget, replace=False)
    rng = np.random.default_rng(seed)
    picks = {"km_random": [], "km_center": [], "km_root": []}
    for label in (0, 1):
        mask = labels[long_ids] == label
        candidates = long_ids[mask]
        groups, distance = cluster_groups(matrix[mask], int(np.sum(labels[reference] == label)), seed)
        for members in groups:
            center = members[np.argmin(distance[members])]
            full = members[roots[candidates[members]]]
            nearest_root = full[np.argmin(distance[full])] if len(full) else center
            picks["km_random"].append(candidates[rng.choice(members)])
            picks["km_center"].append(candidates[center])
            picks["km_root"].append(candidates[nearest_root])
    picks["fragmentation"] = fragmentation(reference, long_ids, labels, words, ratios, seed)
    result = {name: np.random.default_rng(seed).permutation(rows) for name, rows in picks.items()}
    result["len_ge8"] = reference
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", choices=["roots", "pilot"])
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, help="Local copy of SetFit/sst2 train.jsonl")
    args = parser.parse_args()
    pool = load_splits()["pool"]
    ids = np.asarray(pool["orig_idx"], dtype="<i8")
    if hashlib.sha256(ids.tobytes()).hexdigest() != "acf99617135b1c08611e44bb8c9b429f8d64dc3a7d956d17f907fcfd07513698":
        raise ValueError("Pool order differs from the recorded runs")
    source = args.train_jsonl or Path(hf_hub_download("SetFit/sst2", "train.jsonl", repo_type="dataset"))
    texts, labels = list(pool["sentence"]), np.array(pool["label"])
    roots = root_mask(texts, labels, source)
    budget = int(BUDGET_FRACTION * len(pool))
    if args.series == "pilot":
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        words = np.array([len(t.split()) for t in texts])
        matrix, ratios = pilot_features(texts, np.flatnonzero(words >= 8), words)
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        if args.series == "roots":
            subsets, swaps = root_subsets(texts, labels, roots, budget, seed)
            with (args.output / f"pairs_s{seed}.csv").open("w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(["parent_pool_idx", "added_pool_idx", "removed_pool_idx"])
                writer.writerows(swaps)
        else:
            subsets = pilot_subsets(matrix, ratios, labels, words, roots, budget, seed)
        for method, rows in subsets.items():
            np.save(args.output / f"{method}_s{seed}.npy", rows)


if __name__ == "__main__":
    main()
