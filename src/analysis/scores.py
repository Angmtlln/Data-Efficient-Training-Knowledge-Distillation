"""Merge model-free features with training-dynamics / out-of-fold scores into data/scores.parquet.

    python -m src.analysis.scores
"""
import numpy as np
import pandas as pd

from src.config import DATA_DIR, RUNS_DIR

DYNAMICS_FILE = RUNS_DIR / "dynamics_full_seed0.npz"


def dynamics_scores(path=DYNAMICS_FILE):
    d = np.load(path)
    gold = d["gold_prob"]                      # (epochs, n)
    probs1 = np.exp(d["logits"][0]) / np.exp(d["logits"][0]).sum(-1, keepdims=True)
    onehot = np.eye(2)[d["labels"]]
    correct = gold > 0.5
    df = pd.DataFrame(dict(
        pool_idx=d["pool_idx"],
        confidence=gold.mean(0),               # Swayamdipta et al., 2020
        variability=gold.std(0),
        correctness=correct.mean(0),
        el2n=np.linalg.norm(probs1 - onehot, axis=1),   # Paul et al., 2021, after epoch 1
        forgetting=(correct[:-1] & ~correct[1:]).sum(0),
    ))
    return df


def oof_scores(labels):
    path = DATA_DIR / "oof_probs.npy"
    if not path.exists():
        return None
    oof = np.load(path)
    from cleanlab.filter import find_label_issues
    issues = find_label_issues(labels, oof, filter_by="prune_by_noise_rate")
    return pd.DataFrame(dict(oof_gold_prob=oof[np.arange(len(labels)), labels], label_issue=issues))


def main():
    df = pd.read_parquet(DATA_DIR / "features.parquet").merge(dynamics_scores(), on="pool_idx")
    oof = oof_scores(df["label"].to_numpy())
    if oof is not None:
        df = pd.concat([df, oof], axis=1)
    df.to_parquet(DATA_DIR / "scores.parquet")
    print(df.drop(columns=["pool_idx", "cluster"]).describe().round(3).to_string())
    print("\nSpearman correlations between scores:")
    cols = [c for c in ["n_words", "ppl", "confidence", "variability", "el2n", "oof_gold_prob", "test_sim"]
            if c in df]
    print(df[cols].corr("spearman").round(2).to_string())


if __name__ == "__main__":
    main()
