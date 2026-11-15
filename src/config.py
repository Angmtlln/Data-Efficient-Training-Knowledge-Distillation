from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
SUBSETS_DIR = RESULTS_DIR / "subsets"
RUNS_DIR = RESULTS_DIR / "runs"
RESULTS_CSV = RESULTS_DIR / "results.csv"

MODEL_NAME = "answerdotai/ModernBERT-base"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PPL_MODEL_NAME = "openai-community/gpt2"

# SST-2 test labels are hidden, so: GLUE validation -> our test set,
# a stratified slice of GLUE train -> our validation set, rest of train -> selection pool.
VAL_SIZE = 2000
SPLIT_SEED = 42
BUDGET_FRACTION = 0.10

# Fixed hyperparameters shared by every selection strategy.
TRAIN_CFG = dict(
    lr=5e-5,
    batch_size=32,
    epochs=3,
    warmup_ratio=0.06,
    weight_decay=0.01,
    max_length=64,
    eval_batch_size=256,
)
