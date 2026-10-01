"""
plot_confusion.py
─────────────────
Publication-ready confusion-matrix visualisations.

Produces two plot styles per model:
  1. Raw counts (integer heatmap) — good for spotting absolute error magnitudes
  2. Row-normalised percentages — good for spotting per-class recall failures

Both variants are saved to the plots directory. The row-normalised view is
often more revealing on heavily imbalanced datasets because a few raw errors
on a rare class may look negligible in counts but show up as a large
percentage of that class's samples.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_dir: str,
    normalize: bool = True,
) -> str:
    """
    Save a confusion-matrix heatmap for one model.

    Parameters
    ----------
    y_true, y_pred : label arrays
    class_names    : ordered list of human-readable class names
    model_name     : used for the figure title and filename
    output_dir     : where to save the PNG
    normalize      : if True, also save a row-normalised (percentage) heatmap

    Returns
    -------
    str : path to the raw-counts PNG
    """
    os.makedirs(output_dir, exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    count_path = _render_cm(
        cm, class_names, model_name, output_dir,
        is_normalized=False,
        suffix="counts",
    )

    if normalize:
        cm_norm = cm.astype(np.float64)
        row_sums = cm_norm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm_norm, row_sums, out=np.zeros_like(cm_norm), where=row_sums > 0)
        _render_cm(
            cm_norm, class_names, model_name, output_dir,
            is_normalized=True,
            suffix="normalized",
        )

    return count_path


def _render_cm(
    matrix: np.ndarray,
    class_names: list[str],
    model_name: str,
    output_dir: str,
    is_normalized: bool,
    suffix: str,
) -> str:
    """Internal helper — render one heatmap to PNG."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.7)))

    cmap = "Blues"
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=0, vmax=1 if is_normalized else None)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("proportion of true class" if is_normalized else "count")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    ax.set_xlabel("Predicted class", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    title = f"Confusion Matrix — {model_name} ({'row-normalised' if is_normalized else 'counts'})"
    ax.set_title(title, fontweight="bold")

    # Cell annotations — choose text colour based on cell intensity
    if is_normalized:
        fmt = lambda v: f"{v*100:.1f}%"
    else:
        fmt = lambda v: f"{int(v):,}"
    max_val = matrix.max() if matrix.size else 1
    thresh = (max_val / 2.0) if not is_normalized else 0.5
    for i in range(n):
        for j in range(n):
            v = matrix[i, j]
            if is_normalized and v < 0.001:
                continue   # don't clutter with 0.0% on sparse grids
            colour = "white" if v > thresh else "black"
            ax.text(j, i, fmt(v), ha="center", va="center",
                    color=colour, fontsize=8)

    fig.tight_layout()
    fname = f"confusion_{model_name}_{suffix}.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [plot_cm] Saved {path}")
    return path


def plot_all_confusion_matrices_grid(
    confusion_matrices: dict[str, np.ndarray],
    class_names: list[str],
    output_dir: str,
    normalize: bool = True,
) -> str:
    """
    One figure containing all model confusion matrices side-by-side.
    Useful as Figure N in the Results section.
    """
    os.makedirs(output_dir, exist_ok=True)
    models = list(confusion_matrices.keys())
    n_models = len(models)
    n = len(class_names)

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, models):
        cm = confusion_matrices[model_name].astype(np.float64)
        if normalize:
            row_sums = cm.sum(axis=1, keepdims=True)
            cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

        im = ax.imshow(cm, cmap="Blues", aspect="auto", vmin=0, vmax=1 if normalize else None)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=7)
        ax.set_yticklabels(class_names, fontsize=7)
        ax.set_title(model_name, fontweight="bold")
        ax.set_xlabel("Predicted")
        if ax is axes[0]:
            ax.set_ylabel("True")

        # Annotate diagonal only to keep grid readable at this size
        for i in range(n):
            v = cm[i, i]
            txt = f"{v*100:.0f}%" if normalize else f"{int(v):,}"
            colour = "white" if v > (0.5 if normalize else cm.max() / 2) else "black"
            ax.text(i, i, txt, ha="center", va="center", color=colour, fontsize=7)

    fig.suptitle("Confusion Matrices Across Models"
                 + (" (row-normalised)" if normalize else ""),
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = os.path.join(output_dir, "confusion_matrices_grid.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [plot_cm] Saved {path}")
    return path
