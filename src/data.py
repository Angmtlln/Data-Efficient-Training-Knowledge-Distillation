import numpy as np
from datasets import load_from_disk

from src.config import DATA_DIR, SUBSETS_DIR


def load_splits():
    return load_from_disk(str(DATA_DIR / "sst2"))


def save_subset(name: str, indices) -> None:
    SUBSETS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(SUBSETS_DIR / f"{name}.npy", np.asarray(indices, dtype=np.int64))


def load_subset(name: str) -> np.ndarray:
    return np.load(SUBSETS_DIR / f"{name}.npy")
