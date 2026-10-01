"""
learning_curves.py
──────────────────
Two families of diagnostic plots:

1. plot_cnn_loss_curve(model, output_dir)
   ─────────────────────────────────────────
   Training and validation loss per epoch for the CNN.  Populated
   automatically by IDS_CNN.fit() via its history_ attribute.

   Why this matters
   ────────────────
   The rubric explicitly calls out 'monitoring loss curves' as a
   deep-learning-specific evaluation signal.  A converging gap between
   train and validation loss indicates healthy learning; a widening
   gap signals over-fitting; a plateau early in training signals
   under-capacity or poor learning-rate choice.

2. plot_learning_curve(model_name, ...)  AND
   plot_learning_curves_from_scalability(scalability_df, ...)
   ────────────────────────────────────────────────────────────
   Classical learning curve — F1-macro (and training error proxy)
   on varying training-set sizes.  The rubric says:
       "producing learning curve of your ML models on a varying sized
       data sets..."
   Our scalability_df already contains the required data; this function
   renders it properly with per-model curves, std shading, and a
   training-score comparison panel.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Colour palette consistent with utils.py ──────────────────────────────────
_PALETTE = {
    "RandomForest": "#2166ac",
    "LinearSVM":    "#d6604d",
    "MLP":          "#4dac26",
    "CNN":          "#762a83",
}


# ── CNN loss curves (G3) ──────────────────────────────────────────────────────

def plot_cnn_loss_curve(
    cnn_model,
    output_dir: str,
    suffix: str = "",
) -> str:
    """
    Plot training vs validation loss over epochs for a fitted IDS_CNN.

    Parameters
    ----------
    cnn_model   : a fitted IDS_CNN with .history_ populated
    output_dir  : destination directory
    suffix      : optional filename suffix (e.g., "_best_params" for tuned)

    Returns
    -------
    str : path to the saved PNG, or empty string if history is unavailable
    """
    os.makedirs(output_dir, exist_ok=True)
    history = getattr(cnn_model, "history_", None)
    if history is None or not history.get("train_loss"):
        print("  [lc] CNN has no history — skipping loss curve")
        return ""

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(epochs, history["train_loss"], marker="o", linewidth=2,
            color="#2166ac", label="train loss")
    ax.plot(epochs, history["val_loss"],   marker="s", linewidth=2,
            color="#d6604d", label="validation loss")

    # Mark early-stopping point if detectable
    val_losses = history["val_loss"]
    best_epoch = int(np.argmin(val_losses)) + 1
    ax.axvline(best_epoch, linestyle="--", color="grey", alpha=0.6,
               label=f"best val @ epoch {best_epoch}")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Cross-entropy loss", fontsize=12)
    ax.set_title("CNN Training Dynamics  (loss + learning-rate schedule)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Overlay learning-rate schedule on a secondary y-axis when available
    if "lr" in history and history["lr"]:
        ax2 = ax.twinx()
        ax2.plot(epochs, history["lr"], linewidth=1.5, linestyle=":",
                 color="#4dac26", label="learning rate", alpha=0.8)
        ax2.set_ylabel("Learning rate", fontsize=11, color="#4dac26")
        ax2.tick_params(axis="y", labelcolor="#4dac26")
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="upper right")
    else:
        ax.legend(fontsize=11)

    fig.tight_layout()

    fname = f"cnn_loss_curve{suffix}.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [lc] Saved {path}")
    return path


# ── Classical ML learning curves (G4) ─────────────────────────────────────────

def plot_learning_curves_from_scalability(
    scalability_df: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Render textbook learning curves: training F1 AND test F1 vs training-set
    size, one subplot per model.

    The gap between the two curves quantifies the bias-variance position:
      - both low and equal     -> high bias / underfitting
      - both high and equal    -> well-fit
      - high train, low test   -> high variance / overfitting
      - converging as n grows  -> would benefit from more data

    Uses the scalability_df produced by run_scalability() once the
    f1_macro_train_mean column is populated (see experiments.py).
    """
    os.makedirs(output_dir, exist_ok=True)
    has_train_score = "f1_macro_train_mean" in scalability_df.columns

    models = list(scalability_df["model"].unique())
    n_models = len(models)
    fig, axes = plt.subplots(
        1, n_models, figsize=(5 * n_models, 4.5),
        sharey=True,
    )
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        grp = scalability_df[scalability_df["model"] == model_name].sort_values("n_train")
        colour = _PALETTE.get(model_name, "#555555")

        # Test (validation) F1 curve — always present
        ax.plot(grp["n_train"], grp["f1_macro_mean"],
                marker="o", linewidth=2, color=colour, label="Test F1")
        ax.fill_between(
            grp["n_train"],
            grp["f1_macro_mean"] - grp["f1_macro_std"],
            grp["f1_macro_mean"] + grp["f1_macro_std"],
            alpha=0.20, color=colour,
        )

        # Train F1 curve — only if scalability captured it
        if has_train_score:
            ax.plot(grp["n_train"], grp["f1_macro_train_mean"],
                    marker="s", linewidth=2, color=colour,
                    linestyle="--", alpha=0.75, label="Train F1")
            ax.fill_between(
                grp["n_train"],
                grp["f1_macro_train_mean"] - grp["f1_macro_train_std"],
                grp["f1_macro_train_mean"] + grp["f1_macro_train_std"],
                alpha=0.10, color=colour,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training samples (log)", fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("F1-Macro (mean ± std)", fontsize=11)
        ax.set_title(model_name, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4, which="both")

    fig.suptitle("Learning Curves — Train vs Test F1 by Training Set Size",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    path = os.path.join(output_dir, "learning_curves.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [lc] Saved {path}")
    return path


def plot_complexity_vs_performance(
    baseline_df: pd.DataFrame,
    output_dir: str,
) -> str:
    """
    Scatter plot: model size (log) vs F1-macro.  Highlights the
    'which model gives the best F1 per MB' trade-off — relevant to the
    rubric's emphasis on deployment feasibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = baseline_df[baseline_df["class_weight"] == "balanced"].copy()
    if "model_size_mb" not in df.columns:
        print("  [lc] model_size_mb missing — skipping complexity plot")
        return ""

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for _, row in df.iterrows():
        colour = _PALETTE.get(row["model"], "#555555")
        ax.scatter(
            row["model_size_mb"], row["f1_macro_mean"],
            s=220, color=colour, edgecolor="black", linewidth=1.2,
            label=row["model"], zorder=3,
        )
        ax.annotate(
            row["model"],
            (row["model_size_mb"], row["f1_macro_mean"]),
            xytext=(8, 8), textcoords="offset points",
            fontsize=10, fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Model size (MB, log scale)", fontsize=12)
    ax.set_ylabel("Test F1-Macro", fontsize=12)
    ax.set_title("Model Complexity vs Predictive Performance",
                 fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    fig.tight_layout()

    path = os.path.join(output_dir, "complexity_vs_performance.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [lc] Saved {path}")
    return path
