"""Pool-level teacher probabilities for knowledge distillation (src/train.py --teacher <name>).

    python -m src.analysis.teachers
Outputs data/teacher_<name>.npy, shape (n_pool, 2), columns P(negative), P(positive):
  llm  Qwen2.5-3B zero-shot (src/analysis/llm_scores.py), renormalised over positive / negative.
       Uses no labels, so it fits the 10% labeling budget.
  oof  ModernBERT out-of-fold predictions (src/analysis/cv_predict.py): each example is scored by a model trained
       on the other 2/3 of the pool. Sees ~20x more labels than the budget allows: a reference upper bound.
Self-distillation teachers (teacher_self_<subset>.npy) are written by src/train.py --save-probs.
"""
import numpy as np

from src.config import DATA_DIR
from src.data import load_splits


def main():
    labels = np.asarray(load_splits()["pool"]["label"])
    neg, pos, _ = np.exp(np.load(DATA_DIR / "llm_sentiment.npy")).T
    teachers = dict(
        llm=np.stack([neg, pos], 1) / (neg + pos)[:, None],
        oof=np.load(DATA_DIR / "oof_probs.npy"),
    )
    for name, probs in teachers.items():
        probs = probs.astype(np.float32)
        np.save(DATA_DIR / f"teacher_{name}.npy", probs)
        conf = probs.max(1)
        print(f"{name}: accuracy vs gold {(probs.argmax(1) == labels).mean():.4f} | "
              f"mean confidence {conf.mean():.3f} | share with confidence < 0.9: {(conf < 0.9).mean():.3f}")


if __name__ == "__main__":
    main()
