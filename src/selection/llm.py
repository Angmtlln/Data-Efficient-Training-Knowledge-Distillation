"""Selection by LLM-as-judge scores (src/analysis/llm_scores.py), after AlpaGasus (Chen et al., 2024):
filter the data by a judge's rating, then train on what is left.

  llm_quality_top  naive judge: top-k by the 1-5 rating of the labeled example
  llm_top          corrected judge: top-k by judge score = P(gold label) * (1 - P(neutral))
  llm_filter       random among rows with judge score above the median (AlpaGasus keeps a threshold, not a top-k)
  llm_clean        random after removing rows where the zero-shot judge disagrees with the label;
                   the LLM counterpart of the cleanlab filter `clean`
"""
import numpy as np


def judge_score(df):
    return df.llm_gold_prob * (1 - df.llm_neutral)


def llm_quality_top(df, budget, rng):
    s = df.llm_quality.to_numpy() + rng.random(len(df)) * 1e-6
    return np.argsort(s)[-budget:]


def llm_top(df, budget, rng):
    s = judge_score(df).to_numpy() + rng.random(len(df)) * 1e-6
    return np.argsort(s)[-budget:]


def llm_filter(df, budget, rng):
    s = judge_score(df)
    return rng.choice(np.flatnonzero(s >= s.median()), size=budget, replace=False)


def llm_clean(df, budget, rng):
    return rng.choice(np.flatnonzero(df.llm_gold_prob >= 0.5), size=budget, replace=False)
