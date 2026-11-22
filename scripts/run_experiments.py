"""Train on <strategy>_s<seed> subsets one by one, skipping runs already in results.csv.

    python scripts/run_experiments.py --strategies longest kcenter --seeds 0 1 2
    python scripts/run_experiments.py --strategies roots --tag kd_llm -- --teacher llm   # extra args for src.train
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import RESULTS_CSV, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--tag", default="", help="run label; a (subset, seed, tag) triple is trained once")
    p.add_argument("train_args", nargs="*", help="passed to src.train after '--'")
    args = p.parse_args()

    done = set()
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        done = set(zip(df.subset, df.seed, df.tag.fillna("")))

    jobs = [(f"{strat}_s{s}", s) for strat in args.strategies for s in args.seeds]
    for i, (subset, s) in enumerate(jobs, 1):
        if (subset, s, args.tag) in done:
            print(f"[{i}/{len(jobs)}] skip {subset} {args.tag} (already in results.csv)", flush=True)
            continue
        print(f"\n[{i}/{len(jobs)}] === {subset} {args.tag} ===", flush=True)
        # child inherits the terminal, so epoch lines and the progress bar are shown live
        cmd = [sys.executable, "-m", "src.train", "--subset", subset, "--seed", str(s), "--tag", args.tag,
               *args.train_args]
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"FAILED {subset} (exit code {r.returncode})", flush=True)


if __name__ == "__main__":
    main()
