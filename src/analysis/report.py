"""Summary table, subset characteristics, correlations and figures.

    python -m src.analysis.report
Outputs: results/summary.csv, results/subset_features.csv, results/figures/*.png
"""
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_ind

from src.config import DATA_DIR, RESULTS_CSV, RESULTS_DIR, SUBSETS_DIR

FIG_DIR = RESULTS_DIR / "figures"

AXIS = {
    "random": "baseline", "stratified": "baseline", "full": "baseline",
    "longest": "quality", "len_ge8": "quality", "low_ppl": "quality", "clean": "quality",
    "kmeans": "diversity", "kcenter": "diversity",
    "hard": "importance", "easy": "importance", "ambiguous": "importance",
    "hard_bal": "importance", "easy_bal": "importance", "ambiguous_bal": "importance",
    "ccs_b03": "coreset", "ccs_b10": "coreset", "moderate": "coreset", "d2": "coreset", "d2_b10": "coreset",
    "ccs_b10_cheap": "coreset", "ccs_len8": "hybrid", "ccs_len8_cheap": "hybrid",
}
COLORS = {"quality": "#2a78d6", "diversity": "#eb6834", "importance": "#1baf7a", "hybrid": "#6250d6", "coreset": "#e87ba4",
          "baseline": "#8a8985"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def strategy_of(subset):
    return re.sub(r"_s\d+$", "", subset)


def subset_features(scores, idx, emb, test_centroid):
    sub = scores.iloc[idx]
    e = emb[idx]
    return dict(
        mean_words=sub.n_words.mean(),
        short_share=(sub.n_words <= 3).mean(),
        log_ppl=np.log(sub.ppl).mean(),
        pos_share=sub.label.mean(),
        mean_confidence=sub.confidence.mean(),
        noise_share=sub.label_issue.mean() if "label_issue" in sub else np.nan,
        cluster_coverage=sub.cluster.nunique(),
        cluster_entropy=-(lambda p: (p * np.log(p)).sum())(sub.cluster.value_counts(normalize=True)),
        # spread of the subset in embedding space: 1 - |mean embedding|
        embed_spread=1 - np.linalg.norm(e.mean(0)),
        test_sim=float(e.mean(0) @ test_centroid),
    )


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _strategy_panel(ax, s, xlim, refs):
    style(ax)
    y = np.arange(len(s))
    colors = [COLORS[AXIS.get(n, "hybrid")] for n in s.strategy]
    # dot + error bar: bars from an arbitrary zoomed baseline would exaggerate differences
    ax.errorbar(s.test_acc_mean * 100, y, xerr=s.test_acc_std * 100, fmt="none", ecolor=INK2,
                elinewidth=1, capsize=2, zorder=2)
    ax.scatter(s.test_acc_mean * 100, y, s=60, color=colors, edgecolor=SURFACE, linewidth=2, zorder=3)
    for x, label, ls in refs:
        ax.axvline(x * 100, color=INK2, linestyle=ls, linewidth=1, zorder=1)
        if label:
            ax.annotate(label, (x * 100, -0.9), color=INK2, fontsize=8, ha="center", va="top",
                        annotation_clip=False)
    ax.set_yticks(y, s.strategy, color=INK)
    ax.set_ylim(-1.4, len(s) - 0.4)
    ax.set_xlim(*xlim)


def plot_strategies(summary, full_acc, rand_acc):
    s = summary[summary.strategy != "full"].sort_values("test_acc_mean")
    ok, bad = s[s.test_acc_mean > 0.9], s[s.test_acc_mean <= 0.9]
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 0.34 * len(s) + 2.6), facecolor=SURFACE,
        gridspec_kw=dict(height_ratios=[len(ok) + 1.5, len(bad) + 1.5]))
    refs = [(rand_acc, "random", "--"), (full_acc, "full 100%", "-")]
    _strategy_panel(ax1, ok, (rand_acc * 100 - 1.6, full_acc * 100 + 0.5), refs)
    ax1.set_title("Working strategies (zoomed): test accuracy, % — mean ± std over seeds",
                  color=INK, loc="left", fontsize=10)
    _strategy_panel(ax2, bad, (40, 100), [(x, "", ls) for x, _, ls in refs])
    ax2.set_title("Collapsed strategies (full range; dashed = random, solid = full)",
                  color=INK, loc="left", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=c) for c in COLORS.values()]
    ax1.legend(handles, COLORS.keys(), loc="upper left", frameon=False, fontsize=8)
    fig.suptitle("Selection strategies for 10% of SST-2 (ModernBERT-base)", color=INK, x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "strategies.png", dpi=160)
    plt.close(fig)


def plot_correlations(per_strategy, feats):
    n = len(feats)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.1 * rows), facecolor=SURFACE)
    for ax, f in zip(axes.flat, feats):
        style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        d = per_strategy.dropna(subset=[f])
        for _, r in d.iterrows():
            ax.scatter(r[f], r.test_acc * 100, s=40, color=COLORS[AXIS.get(r.strategy, "hybrid")],
                       edgecolor=SURFACE, linewidth=1, zorder=3)
        rho, p = spearmanr(d[f], d.test_acc)
        ax.set_title(f"{f}   ρ={rho:+.2f} (p={p:.2g})", color=INK, fontsize=9, loc="left")
        ax.set_ylabel("test acc, %", color=INK2, fontsize=8)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle("Subset characteristics vs test accuracy (one point = strategy mean, working strategies only)",
                 color=INK, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlations.png", dpi=160)
    plt.close(fig)


def plot_cartography(scores):
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    sample = scores.sample(min(15000, len(scores)), random_state=0)
    noisy = sample.label_issue if "label_issue" in sample else np.zeros(len(sample), bool)
    ax.scatter(sample.variability[~noisy], sample.confidence[~noisy], s=3, alpha=0.3,
               color=COLORS["quality"], label="clean", rasterized=True)
    ax.scatter(sample.variability[noisy], sample.confidence[noisy], s=5, alpha=0.6,
               color=COLORS["diversity"], label="label issue (cleanlab)", rasterized=True)
    ax.set_xlabel("variability", color=INK2)
    ax.set_ylabel("confidence", color=INK2)
    ax.legend(frameon=False, fontsize=8, markerscale=3)
    ax.set_title("Dataset cartography of the SST-2 pool", color=INK, loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cartography.png", dpi=160)
    plt.close(fig)


def plot_val_vs_test(summary, label=("len_ge8", "ccs_len8", "ccs_b10", "ccs_b10_cheap", "d2_b10",
                                      "longest", "rank_agg", "kcenter", "random")):
    s = summary[(summary.test_acc_mean > 0.9) & (summary.strategy != "full")]
    fig, ax = plt.subplots(figsize=(6.5, 5.2), facecolor=SURFACE)
    style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for _, r in s.iterrows():
        ax.scatter(r.val_acc_mean * 100, r.test_acc_mean * 100, s=60, zorder=3,
                   color=COLORS[AXIS.get(r.strategy, "hybrid")], edgecolor=SURFACE, linewidth=2)
        if r.strategy in label:
            left = r.strategy == "ccs_len8"  # sits right next to len_ge8
            ax.annotate(r.strategy, (r.val_acc_mean * 100, r.test_acc_mean * 100),
                        xytext=(-6, -10) if left else (6, 3), ha="right" if left else "left",
                        textcoords="offset points", fontsize=8, color=INK2)
    rho = spearmanr(s.val_acc_mean, s.test_acc_mean).statistic
    ax.set_xlabel("val accuracy, % (phrases, same distribution as train)", color=INK2)
    ax.set_ylabel("test accuracy, % (full sentences)", color=INK2)
    ax.set_title(f"Strategy ranking on val vs test (Spearman ρ = {rho:+.2f})", color=INK, loc="left", fontsize=11)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=7, color=c) for c in COLORS.values()]
    ax.legend(handles, COLORS.keys(), loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "val_vs_test.png", dpi=160)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(RESULTS_CSV)
    res["strategy"] = res.subset.map(strategy_of)

    summary = res.groupby("strategy").agg(
        n_runs=("seed", "size"),
        test_acc_mean=("test_acc", "mean"), test_acc_std=("test_acc", "std"),
        test_f1_mean=("test_f1", "mean"), test_f1_std=("test_f1", "std"),
        val_acc_mean=("val_acc", "mean"), val_acc_std=("val_acc", "std"),
    ).reset_index()
    full_acc = summary.set_index("strategy").test_acc_mean["full"]
    rand_acc = summary.set_index("strategy").test_acc_mean["random"]
    summary["delta_vs_random_pp"] = (summary.test_acc_mean - rand_acc) * 100
    summary["gap_closed_%"] = (summary.test_acc_mean - rand_acc) / (full_acc - rand_acc) * 100
    rand_runs = res[res.strategy == "random"].test_acc
    summary["p_vs_random"] = summary.strategy.map(
        lambda n: ttest_ind(res[res.strategy == n].test_acc, rand_runs, equal_var=False).pvalue
        if n not in ("random", "full") else np.nan)
    rand_val = res[res.strategy == "random"].val_acc
    summary["p_val_vs_random"] = summary.strategy.map(
        lambda n: ttest_ind(res[res.strategy == n].val_acc, rand_val, equal_var=False).pvalue
        if n not in ("random", "full") else np.nan)
    summary["axis"] = summary.strategy.map(lambda n: AXIS.get(n, "hybrid"))
    summary = summary.sort_values("test_acc_mean", ascending=False)
    summary.to_csv(RESULTS_DIR / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))

    scores = pd.read_parquet(DATA_DIR / "scores.parquet")
    emb = np.load(DATA_DIR / "emb_pool.npy").astype(np.float32)
    emb_test = np.load(DATA_DIR / "emb_test.npy").astype(np.float32)
    test_centroid = emb_test.mean(0) / np.linalg.norm(emb_test.mean(0))

    rows = []
    for subset in res.subset.unique():
        if subset == "full":
            continue
        idx = np.load(SUBSETS_DIR / f"{subset}.npy")
        rows.append(dict(subset=subset, **subset_features(scores, idx, emb, test_centroid)))
    feats_df = pd.DataFrame(rows)
    per_run = res.merge(feats_df, on="subset")
    feats = [c for c in feats_df.columns if c != "subset"]
    per_strategy = per_run.groupby("strategy")[feats + ["test_acc"]].mean().reset_index()
    per_strategy.to_csv(RESULTS_DIR / "subset_features.csv", index=False)

    print("\nSpearman: subset characteristic vs test accuracy")
    # collapsed strategies (acc ~ chance) dominate rank correlations; also look among working subsets only
    working = per_run[per_run.test_acc > 0.9]
    for level, d in (("per strategy", per_strategy), ("per run", per_run), ("per run, acc>0.9", working)):
        corr = {f: spearmanr(d[f], d.test_acc, nan_policy="omit") for f in feats}
        print(f"  {level} (n={len(d)}): " + ", ".join(f"{f} {c.statistic:+.2f}" for f, c in corr.items()))

    plot_strategies(summary, full_acc, rand_acc)
    plot_correlations(per_strategy[per_strategy.test_acc > 0.9], feats)
    plot_cartography(scores)
    plot_val_vs_test(summary)
    print(f"\nfigures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
