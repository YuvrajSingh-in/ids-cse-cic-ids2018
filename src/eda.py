"""
eda.py
──────
Exploratory Data Analysis for the CSE-CIC-IDS2018 dataset.

Produces
────────
1. descriptive_stats.csv      — per-feature count/mean/std/min/max/quartiles
2. class_distribution.csv     — class counts, percentages, imbalance ratios
3. class_distribution.png     — bar chart (absolute + log scale)
4. missing_value_summary.txt  — rows lost to NaN/Inf per source file
5. correlation_heatmap.png    — top-20 features by variance, pairwise correlations
6. feature_variance.csv       — ranked feature variances (low-variance candidates for pruning)

Design
──────
- All statistics are computed on the TRAINING feature matrix only.
  EDA must never touch the test set — doing so constitutes data leakage
  and inflates downstream performance estimates.
- Correlation heatmap is restricted to the top-20 highest-variance features.
  A full 78x78 heatmap is unreadable and provides little insight.
- Class imbalance is summarised both as counts and as the ratio of the
  most-populous class to each other class.  The IDS literature consistently
  flags this ratio as the primary driver of classifier bias.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── Descriptive statistics ────────────────────────────────────────────────────

def descriptive_statistics(
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str,
) -> pd.DataFrame:
    """
    Compute per-feature descriptive statistics on the training matrix.

    Columns: count, mean, std, min, 25%, 50%, 75%, max, skew, kurt
    Saved as descriptive_stats.csv.
    """
    _ensure_dir(output_dir)
    df = pd.DataFrame(X, columns=feature_names)
    # pandas.describe gives count/mean/std/min/quartiles/max
    desc = df.describe().T
    # Add shape statistics that often reveal heavy-tailed flow features
    desc["skew"] = df.skew().values
    desc["kurt"] = df.kurt().values
    out = os.path.join(output_dir, "descriptive_stats.csv")
    desc.to_csv(out)
    print(f"  [eda] Saved {out}  ({len(desc)} features)")
    return desc


# ── Class distribution ────────────────────────────────────────────────────────

def class_distribution(
    y: np.ndarray,
    class_names: list[str],
    output_dir: str,
) -> pd.DataFrame:
    """
    Tabulate and plot class counts, proportions, and imbalance ratios.
    Imbalance ratio = max_class_count / this_class_count (>=1, larger = rarer).
    """
    _ensure_dir(output_dir)
    unique, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    max_count = counts.max()
    rows = []
    for u, c in zip(unique, counts):
        rows.append({
            "class_index":     int(u),
            "class_name":      class_names[u],
            "count":           int(c),
            "percentage":      round(100 * c / total, 3),
            "imbalance_ratio": round(max_count / c, 2),
        })
    df = pd.DataFrame(rows).sort_values("count", ascending=False)
    out_csv = os.path.join(output_dir, "class_distribution.csv")
    df.to_csv(out_csv, index=False)
    print(f"  [eda] Saved {out_csv}")

    # Plot — linear + log scale in one figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colours = ["#2166ac"] + ["#d6604d"] * (len(df) - 1)   # highlight majority

    for ax, scale in zip(axes, ("linear", "log")):
        ax.bar(df["class_name"], df["count"], color=colours, edgecolor="black", alpha=0.85)
        ax.set_yscale(scale)
        ax.set_title(f"Class Distribution ({scale} scale)", fontweight="bold")
        ax.set_ylabel("Number of flows")
        ax.tick_params(axis="x", rotation=25)
        for label in ax.get_xticklabels():
            label.set_ha("right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(f"CSE-CIC-IDS2018 — Class Distribution (n={total:,})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_png = os.path.join(output_dir, "class_distribution.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  [eda] Saved {out_png}")
    return df


# ── Feature variance ──────────────────────────────────────────────────────────

def feature_variance_report(
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str,
) -> pd.DataFrame:
    """
    Rank features by variance.  Near-zero-variance features contribute
    essentially nothing to a classifier and are candidates for removal.
    """
    _ensure_dir(output_dir)
    variances = X.var(axis=0)
    df = pd.DataFrame({
        "feature":  feature_names,
        "variance": variances,
    }).sort_values("variance", ascending=False).reset_index(drop=True)
    out = os.path.join(output_dir, "feature_variance.csv")
    df.to_csv(out, index=False)

    # Flag near-zero-variance feats (useful for report commentary)
    near_zero = df[df["variance"] < 1e-6]
    print(f"  [eda] Saved {out}")
    print(f"  [eda] {len(near_zero)} near-zero-variance features "
          f"(variance < 1e-6) — candidates for removal")
    return df


# ── Correlation heatmap ───────────────────────────────────────────────────────

def correlation_heatmap(
    X: np.ndarray,
    feature_names: list[str],
    output_dir: str,
    top_k: int = 20,
) -> pd.DataFrame:
    """
    Pairwise Pearson correlations on the top-k highest-variance features.

    Full 78x78 heatmaps are unreadable; focusing on the features most likely
    to be informative (highest variance) gives a compact view of collinearity.
    Highly correlated feature pairs are candidates for removal or PCA.
    """
    _ensure_dir(output_dir)
    variances = X.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:top_k]
    sub_X = X[:, top_idx]
    sub_names = [feature_names[i] for i in top_idx]

    corr = np.corrcoef(sub_X, rowvar=False)
    corr_df = pd.DataFrame(corr, index=sub_names, columns=sub_names)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(top_k))
    ax.set_yticks(range(top_k))
    ax.set_xticklabels(sub_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(sub_names, fontsize=8)
    ax.set_title(f"Pearson Correlation — Top {top_k} Features by Variance",
                 fontweight="bold")
    plt.colorbar(im, ax=ax, label="correlation", shrink=0.8)
    fig.tight_layout()
    out_png = os.path.join(output_dir, "correlation_heatmap.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"  [eda] Saved {out_png}")

    # Report highly correlated pairs (excluding diagonal)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    high_corr_pairs = []
    for i, j in zip(*np.where(mask & (np.abs(corr) > 0.9))):
        high_corr_pairs.append((sub_names[i], sub_names[j], corr[i, j]))
    if high_corr_pairs:
        print(f"  [eda] {len(high_corr_pairs)} feature pairs with |r|>0.9 "
              "— evidence of redundancy (candidates for PCA/selection)")
    return corr_df


# ── Orchestrator ──────────────────────────────────────────────────────────────

def run_eda(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    output_dir: str,
) -> dict:
    """
    Run all EDA steps. Returns a dict of the produced DataFrames for
    downstream use or notebook inspection.
    """
    print(f"\n  [eda] Running EDA on training set  "
          f"({X_train.shape[0]:,} rows × {X_train.shape[1]} feats)")
    _ensure_dir(output_dir)

    return {
        "descriptive":  descriptive_statistics(X_train, feature_names, output_dir),
        "class_dist":   class_distribution(y_train, class_names, output_dir),
        "variance":     feature_variance_report(X_train, feature_names, output_dir),
        "correlation":  correlation_heatmap(X_train, feature_names, output_dir, top_k=20),
    }
