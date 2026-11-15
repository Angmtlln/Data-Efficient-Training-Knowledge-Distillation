"""Download GLUE SST-2 and build pool / val / test splits under data/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import DatasetDict, load_dataset

from src.config import BUDGET_FRACTION, DATA_DIR, SPLIT_SEED, VAL_SIZE


def main():
    raw = load_dataset("nyu-mll/glue", "sst2")
    train = raw["train"].rename_column("idx", "orig_idx")
    test = raw["validation"].rename_column("idx", "orig_idx")

    split = train.train_test_split(
        test_size=VAL_SIZE, seed=SPLIT_SEED, stratify_by_column="label"
    )
    pool, val = split["train"], split["test"]
    # pool_idx is the stable id used by every selection strategy
    pool = pool.add_column("pool_idx", list(range(len(pool))))

    ds = DatasetDict(pool=pool, val=val, test=test)
    ds.save_to_disk(str(DATA_DIR / "sst2"))

    for name, d in ds.items():
        pos = sum(d["label"]) / len(d)
        print(f"{name:5s}: {len(d):6d} examples, positive share {pos:.3f}")
    print(f"10% budget = {int(len(pool) * BUDGET_FRACTION)} examples")


if __name__ == "__main__":
    main()
