"""Coreset methods from the literature, adapted to the SST-2 pool.

- CCS: Zheng et al., "Coverage-centric Coreset Selection for High Pruning Rates", ICLR 2023
- Moderate: Xia et al., "Moderate Coreset", ICLR 2023
- D2 Pruning: Maharana et al., "D2 Pruning: Message Passing for Balancing Diversity & Difficulty", ICLR 2024

Difficulty score = EL2N after epoch 1 of the full-data run (higher = harder).
Embeddings = MiniLM sentence embeddings (data/emb_pool.npy), indexed by pool_idx.
"""
import numpy as np
import torch

from src.config import DATA_DIR

_EMB = None


def _embeddings(df):
    global _EMB
    if _EMB is None:
        _EMB = np.load(DATA_DIR / "emb_pool.npy").astype(np.float32)
    return _EMB[df.pool_idx.to_numpy()]


def ccs(beta, n_strata=50, score="el2n"):
    """Prune the hardest `beta` fraction, split the rest into equal-width score strata and
    spread the budget over strata as evenly as possible (smallest strata are taken whole)."""
    def select(df, budget, rng):
        s = df[score].to_numpy()
        order = np.argsort(s)
        keep = order[: len(s) - int(beta * len(s))]
        edges = np.linspace(s[keep].min(), s[keep].max(), n_strata + 1)
        strata = np.clip(np.digitize(s[keep], edges[1:-1]), 0, n_strata - 1)
        groups = sorted((keep[strata == k] for k in range(n_strata)), key=len)
        groups = [g for g in groups if len(g)]
        out, left = [], budget
        for i, g in enumerate(groups):
            take = min(len(g), left // (len(groups) - i))
            out.append(rng.choice(g, size=take, replace=False))
            left -= take
        return np.concatenate(out)
    return select


def moderate(df, budget, rng):
    """Per class: keep examples whose distance to the class centroid is closest to the median distance."""
    x = _embeddings(df)
    labels = df.label.to_numpy()
    out = []
    for c in np.unique(labels):
        cls = np.flatnonzero(labels == c)
        d = np.linalg.norm(x[cls] - x[cls].mean(0), axis=1)
        k = round(budget * len(cls) / len(labels))
        out.append(cls[np.argsort(np.abs(d - np.median(d)))[:k]])
    return np.concatenate(out)[:budget]


def _knn(x, k, batch=4096):
    x = torch.from_numpy(x).cuda()
    dists, idxs = [], []
    for i in range(0, len(x), batch):
        d = torch.cdist(x[i:i + batch], x)
        d[torch.arange(d.shape[0]), torch.arange(i, i + d.shape[0])] = float("inf")  # no self-loops
        v, j = d.topk(k, largest=False)
        dists.append(v.cpu())
        idxs.append(j.cpu())
    return torch.cat(dists).numpy(), torch.cat(idxs).numpy()


def d2(beta=0.0, k=10, gamma_f=1.0, gamma_r=1.0, score="el2n"):
    """Forward message passing spreads difficulty over a kNN graph; greedy selection with
    reverse message passing down-weights neighbours of every selected node."""
    def select(df, budget, rng):
        n = len(df)
        s = df[score].to_numpy().astype(np.float64)
        s = (s - s.min()) / (s.max() - s.min())
        dist, nbr = _knn(_embeddings(df), k)
        w_f, w_r = np.exp(-gamma_f * dist), np.exp(-gamma_r * dist)
        node = s + (w_f * s[nbr]).sum(1)
        if beta > 0:  # prune the hardest beta fraction, as in CCS
            node[np.argsort(s)[n - int(beta * n):]] = -np.inf
        node += rng.random(n) * 1e-9
        out = []
        for _ in range(budget):
            i = int(np.argmax(node))
            out.append(i)
            node[nbr[i]] -= w_r[i] * node[i]   # already selected nodes stay at -inf
            node[i] = -np.inf
        return np.array(out)
    return select
