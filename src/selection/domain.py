"""Domain-classifier selection (discriminative importance resampling, as in the GPT-3 quality filter,
Brown et al., 2020): logistic regression on MiniLM embeddings separates unlabeled target sentences
(GLUE SST-2 test split, labels hidden) from pool examples; pool examples are then sampled by the
classifier's log-odds with Gumbel top-k.
"""
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from src.config import DATA_DIR, EMBED_MODEL_NAME

_CACHE = {}


def _target_embeddings():
    path = DATA_DIR / "emb_target.npy"
    if not path.exists():
        texts = load_dataset("nyu-mll/glue", "sst2")["test"]["sentence"]
        emb = SentenceTransformer(EMBED_MODEL_NAME, device="cuda").encode(
            texts, batch_size=512, normalize_embeddings=True)
        np.save(path, emb.astype(np.float16))
    return np.load(path).astype(np.float32)


def _log_odds():
    if "log_odds" not in _CACHE:
        pool = np.load(DATA_DIR / "emb_pool.npy").astype(np.float32)
        target = _target_embeddings()
        neg = np.random.default_rng(0).choice(len(pool), size=len(target) * 5, replace=False)
        x = np.concatenate([target, pool[neg]])
        y = np.r_[np.ones(len(target)), np.zeros(len(neg))]
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
        auc = cross_val_score(clf, x, y, cv=5, scoring="roc_auc").mean()
        print(f"domain classifier: 5-fold AUC target vs pool = {auc:.3f}")
        _CACHE["log_odds"] = clf.fit(x, y).decision_function(pool)
    return _CACHE["log_odds"]


def domain_clf(df, budget, rng):
    s = _log_odds()[df.pool_idx.to_numpy()]
    return np.argsort(-(s + rng.gumbel(size=len(s))))[:budget]
