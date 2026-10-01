"""
utils.py
────────
Shared utilities: plot generation, result printing, CSV persistence,
and research-level model selection summary.

All plots are publication-ready:
- Axes labelled with units
- Error bars (±1 std) where applicable
- Consistent colour palette across figures
- Saved as high-resolution PNG (300 dpi)

Matplotlib is used (no seaborn dependency) so the project stays
lightweight and reproducible across environments.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe in any env
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Consistent colour palette ─────────────────────────────────────────────────
_PALETTE = {
    "RandomForest": "#2166ac",   # deep blue
    "LinearSVM":    "#d6604d",   # muted red
    "MLP":          "#4dac26",   # forest green
}
_FALLBACK_COLOURS = ["#8073ac", "#e08214", "#762a83"]


# ── Plot 1: F1 vs Dataset Size ────────────────────────────────────────────────

def plot_f1_vs_size(
    scalability_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Line plot showing how F1-macro evolves as training set grows."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, grp in scalability_df.groupby("model"):
        grp = grp.sort_values("fraction")
        colour = _PALETTE.get(model_name, "#555555")
        ax.plot(
            grp["fraction"] * 100,
            grp["f1_macro_mean"],
            marker="o",
            linewidth=2,
            color=colour,
            label=model_name,
        )
        ax.fill_between(
            grp["fraction"] * 100,
            grp["f1_macro_mean"] - grp["f1_macro_std"],
            grp["f1_macro_mean"] + grp["f1_macro_std"],
            alpha=0.15,
            color=colour,
        )

    ax.set_xlabel("Training Set Size (%)", fontsize=12)
    ax.set_ylabel("F1-Macro (mean ± std)", fontsize=12)
    ax.set_title("F1-Macro Score vs Training Dataset Size", fontsize=13, fontweight="bold")
    ax.set_xlim(20, 105)
    ax.set_ylim(0, 1.05)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(output_dir, "f1_vs_size.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [plot] Saved: {path}")


# ── Plot 2: Training Time vs Dataset Size ─────────────────────────────────────

def plot_train_time_vs_size(
    scalability_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Line plot of training time (seconds) as training set grows."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, grp in scalability_df.groupby("model"):
        grp = grp.sort_values("fraction")
        colour = _PALETTE.get(model_name, "#555555")
        ax.plot(
            grp["fraction"] * 100,
            grp["train_time_mean"],
            marker="s",
            linewidth=2,
            color=colour,
            label=model_name,
        )
        ax.fill_between(
            grp["fraction"] * 100,
            grp["train_time_mean"] - grp["train_time_std"],
            grp["train_time_mean"] + grp["train_time_std"],
            alpha=0.15,
            color=colour,
        )

    ax.set_xlabel("Training Set Size (%)", fontsize=12)
    ax.set_ylabel("Training Time (seconds)", fontsize=12)
    ax.set_title("Training Time vs Training Dataset Size", fontsize=13, fontweight="bold")
    ax.set_xlim(20, 105)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(output_dir, "train_time_vs_size.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [plot] Saved: {path}")


# ── Plot 3: Model Comparison Bar Chart (Baseline) ────────────────────────────

def plot_model_comparison(
    baseline_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    Grouped bar chart comparing F1-macro and F1-weighted for all models.
    Error bars represent ±1 std across 3 runs.
    Only rows with class_weight='balanced' are shown (primary result).
    """
    df = baseline_df[baseline_df["class_weight"] == "balanced"].copy()

    models = df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x - width / 2,
        df["f1_macro_mean"],
        width,
        yerr=df["f1_macro_std"],
        label="F1-Macro",
        color=[_PALETTE.get(m, "#888888") for m in models],
        alpha=0.85,
        capsize=4,
    )
    bars2 = ax.bar(
        x + width / 2,
        df["f1_weighted_mean"],
        width,
        yerr=df["f1_weighted_std"],
        label="F1-Weighted",
        color=[_PALETTE.get(m, "#888888") for m in models],
        alpha=0.45,
        capsize=4,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("F1 Score (mean ± std)", fontsize=12)
    ax.set_title("Model Comparison — Baseline (Full Dataset, Balanced Weights)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    # Annotate bar tops
    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.01,
            f"{h:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    fig.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [plot] Saved: {path}")


# ── Plot 4: Class Weight Comparison ──────────────────────────────────────────

def plot_class_weight_comparison(
    cw_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Grouped bar chart: None vs balanced, per model."""
    models = cw_df["model"].unique()
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, cw in enumerate([None, "balanced"]):
        subset = cw_df[cw_df["class_weight"] == str(cw)]
        vals = [
            subset[subset["model"] == m]["f1_macro_mean"].values[0]
            if m in subset["model"].values else 0.0
            for m in models
        ]
        errs = [
            subset[subset["model"] == m]["f1_macro_std"].values[0]
            if m in subset["model"].values else 0.0
            for m in models
        ]
        offset = (i - 0.5) * width
        ax.bar(
            x + offset, vals, width,
            yerr=errs,
            label=f"class_weight={cw!r}",
            alpha=0.8, capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("F1-Macro (mean ± std)", fontsize=12)
    ax.set_title("Impact of Class Weighting on F1-Macro",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    path = os.path.join(output_dir, "class_weight_comparison.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [plot] Saved: {path}")


# ── Console Tables ────────────────────────────────────────────────────────────

def print_baseline_table(baseline_df: pd.DataFrame) -> None:
    """Render a clean summary table to stdout (balanced rows only)."""
    df = baseline_df[baseline_df["class_weight"] == "balanced"]
    sep = "─" * 130
    print(f"\n{sep}")
    print("  BASELINE RESULTS  (class_weight=balanced, multi-run mean ± std)")
    print(sep)
    has_auc  = "auc_roc_mean"  in df.columns
    has_size = "model_size_mb" in df.columns

    header = (
        f"  {'Model':<14} {'Precision':>10} {'Recall':>10} "
        f"{'F1-Macro':>14} {'F1-Wtd':>14} "
    )
    if has_auc:  header += f"{'AUC-ROC':>9} "
    header += f"{'Train(s)':>10} {'Lat(ms)':>9} {'Tput(/s)':>10} "
    if has_size: header += f"{'Size(MB)':>9}"
    print(header)
    print(sep)

    for _, row in df.iterrows():
        line = (
            f"  {row['model']:<14} "
            f"{row['precision_mean']:>10.4f} "
            f"{row['recall_mean']:>10.4f} "
            f"  {row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f} "
            f"  {row['f1_weighted_mean']:.4f}±{row['f1_weighted_std']:.4f} "
        )
        if has_auc:
            auc = row["auc_roc_mean"]
            line += f"{auc:>9.4f} " if not pd.isna(auc) else f"{'n/a':>9} "
        line += (
            f"{row['train_time_mean']:>9.2f}s "
            f"{row['latency_ms_mean']:>8.3f}ms "
            f"{row['throughput_mean']:>9.0f}/s "
        )
        if has_size:
            line += f"{row['model_size_mb']:>9.2f}"
        print(line)
    print(sep)


def print_scalability_table(scalability_df: pd.DataFrame) -> None:
    """Pivot F1-macro mean by fraction and model."""
    pivot_f1 = scalability_df.pivot_table(
        index="fraction", columns="model", values="f1_macro_mean"
    )
    pivot_t = scalability_df.pivot_table(
        index="fraction", columns="model", values="train_time_mean"
    )
    pivot_f1.index = [f"{int(f*100)}%" for f in pivot_f1.index]
    pivot_t.index  = [f"{int(f*100)}%" for f in pivot_t.index]

    print("\n  SCALABILITY — F1-Macro (mean)")
    print(pivot_f1.round(4).to_string())
    print("\n  SCALABILITY — Training Time in seconds (mean)")
    print(pivot_t.round(2).to_string())


def print_class_weight_table(cw_df: pd.DataFrame) -> None:
    """Side-by-side F1 comparison for class_weight=None vs balanced."""
    pivot = cw_df.pivot_table(
        index="model", columns="class_weight",
        values=["f1_macro_mean", "f1_macro_std"],
    )
    print("\n  CLASS WEIGHT STUDY — F1-Macro (mean)")
    print(pivot.round(4).to_string())


# ── Research Summary ──────────────────────────────────────────────────────────

def print_research_summary(baseline_df: pd.DataFrame) -> None:
    """
    Select and justify the best model along three research dimensions:
    accuracy, speed, and the accuracy-latency trade-off.

    This summary targets an examiner audience — concise, evidence-based,
    no generic platitudes.
    """
    df = baseline_df[baseline_df["class_weight"] == "balanced"].copy()
    sep = "═" * 70

    best_f1_row    = df.loc[df["f1_macro_mean"].idxmax()]
    best_speed_row = df.loc[df["throughput_mean"].idxmax()]

    # Trade-off: normalise both metrics 0→1, maximise their harmonic mean
    df["f1_norm"]   = df["f1_macro_mean"] / df["f1_macro_mean"].max()
    df["tput_norm"] = df["throughput_mean"] / df["throughput_mean"].max()
    df["tradeoff"]  = 2 * (df["f1_norm"] * df["tput_norm"]) / (df["f1_norm"] + df["tput_norm"])
    best_tradeoff_row = df.loc[df["tradeoff"].idxmax()]

    # Per-model explanatory hints — selected dynamically based on which
    # model actually wins each criterion. Avoids the original bug where
    # the text was hardcoded to RF / LinearSVM regardless of result.
    accuracy_rationale = {
        "RandomForest":
            "Random Forest captures non-linear feature boundaries without sensitivity\n"
            "     to feature scale — a decisive advantage on the heterogeneous IDS2018\n"
            "     feature space (flow statistics spanning 10+ orders of magnitude).",
        "LinearSVM":
            "LinearSVM's hyperplane separates the dominant attack signatures\n"
            "     well after StandardScaler normalisation; the largely linear class\n"
            "     boundaries on flow features explain the strong macro-F1.",
        "MLP":
            "The MLP learns a non-linear feature combination via two hidden layers\n"
            "     and benefits from balanced sampling implicit in mini-batch training,\n"
            "     yielding the strongest macro-F1 across all six attack classes.",
        "CNN":
            "The 1D-CNN exploits a convolutional inductive bias over adjacent\n"
            "     flow-statistic features and uses class-weighted CrossEntropy to\n"
            "     focus learning on rare attack classes.",
    }
    speed_rationale = {
        "RandomForest":
            "Random Forest's vectorised tree traversal makes batched inference\n"
            "     fast on modern CPUs, achieving high throughput despite the model\n"
            "     ensemble size.",
        "LinearSVM":
            "LinearSVM's single hyperplane evaluates in O(d) per sample, making it\n"
            "     the only candidate with credible real-time detection throughput at\n"
            "     line rate.",
        "MLP":
            "The compact MLP (128, 64 hidden units) requires only two matrix\n"
            "     multiplications per inference, giving it competitive throughput.",
        "CNN":
            "The CNN's small filter banks and pooling layers keep the per-sample\n"
            "     compute bounded; with batched inference on GPU it can sustain very\n"
            "     high throughput.",
    }

    print(f"\n{sep}")
    print("  RESEARCH SUMMARY — Model Selection Analysis")
    print(sep)
    print(
        f"\n  🏆 Best Accuracy:   {best_f1_row['model']}"
        f"\n     F1-Macro = {best_f1_row['f1_macro_mean']:.4f} ± {best_f1_row['f1_macro_std']:.4f}"
        f"\n     {accuracy_rationale.get(best_f1_row['model'], '')}"
    )
    print(
        f"\n  ⚡ Best Speed:      {best_speed_row['model']}"
        f"\n     Throughput = {best_speed_row['throughput_mean']:.0f} samples/s"
        f"  |  Latency = {best_speed_row['latency_ms_mean']:.3f} ms/sample"
        f"\n     {speed_rationale.get(best_speed_row['model'], '')}"
    )
    print(
        f"\n  ⚖️  Best Trade-off:  {best_tradeoff_row['model']}"
        f"\n     Trade-off Score = {best_tradeoff_row['tradeoff']:.4f}"
        f"\n     (harmonic mean of normalised F1 and throughput)"
        f"\n     This model offers the best operationally deployable balance"
        f"\n     between detection quality and inference speed for a production IDS."
    )
    print(f"\n{sep}\n")


# ── Persistence ───────────────────────────────────────────────────────────────

def save_results(
    baseline_df: pd.DataFrame,
    scalability_df: pd.DataFrame,
    cw_df: pd.DataFrame,
    results_dir: str,
) -> None:
    """Write all experiment DataFrames to CSV in the results directory."""
    os.makedirs(results_dir, exist_ok=True)
    paths = {
        "baseline_results.csv":       baseline_df,
        "scalability_results.csv":    scalability_df,
        "class_weight_comparison.csv": cw_df,
    }
    for filename, df in paths.items():
        path = os.path.join(results_dir, filename)
        df.to_csv(path, index=False)
        print(f"  [save] {path}")
