"""DSIR: Xie et al., "Data Selection for Language Models via Importance Resampling", NeurIPS 2023.

Target distribution = unlabeled sentences of the GLUE SST-2 test split (labels are hidden; our
evaluation set is GLUE validation, so nothing we score on is used for selection).
Bag of hashed uni+bigrams for target and raw (pool) -> log importance weight per example ->
Gumbel top-k sampling without replacement.
"""
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer

from src.data import load_splits

_CACHE = {}


def _log_weights():
    if "logw" not in _CACHE:
        # murmurhash-based, so features are identical across processes (unlike Python's hash())
        vec = HashingVectorizer(ngram_range=(1, 2), n_features=10_000, alternate_sign=False, norm=None,
                                token_pattern=r"\S+", lowercase=True)
        pool = vec.transform(load_splits()["pool"]["sentence"])
        target = vec.transform(load_dataset("nyu-mll/glue", "sst2")["test"]["sentence"])
        log_p = lambda m: np.log((np.asarray(m.sum(0)).ravel() + 1) / (m.sum() + m.shape[1]))
        _CACHE["logw"] = np.asarray(pool @ (log_p(target) - log_p(pool))).ravel()
    return _CACHE["logw"]


def dsir(df, budget, rng):
    logw = _log_weights()[df.pool_idx.to_numpy()]
    return np.argsort(-(logw + rng.gumbel(size=len(logw))))[:budget]
