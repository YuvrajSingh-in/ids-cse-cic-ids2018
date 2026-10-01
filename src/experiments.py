"""
experiments.py
──────────────
Orchestrates all experimental loops.

Three experiment types are implemented:

1. run_baseline()
   ─────────────
   Train and evaluate all models on the FULL training set.
   Repeated n_runs times (default 3) to produce statistically
   meaningful mean ± std estimates rather than a single point.

2. run_scalability()
   ──────────────────
   Train each model on progressively larger subsets of the training
   data (25%, 50%, 75%, 100%).  Tracks how F1, training time, latency,
   and throughput evolve — key for assessing deployment feasibility.

3. run_class_weight_study()
   ─────────────────────────
   Compares class_weight=None vs class_weight="balanced" across all
   models that support the parameter.  Isolates the contribution of
   imbalance handling to model performance — a standard research
   ablation in IDS literature.

Data routing convention (CRITICAL)
────────────────────────────────────
  Random Forest → raw (unscaled) features: X_train_raw / X_test_raw
  LinearSVM     → scaled features:         X_train_s   / X_test_s
  MLP           → scaled features:         X_train_s   / X_test_s

This routing is enforced inside _select_features() below.
"""

import copy
import numpy as np
import pandas as pd
from collections import defaultdict

from src.models import get_models
from src.evaluation import train_model, evaluate_model

RANDOM_STATE = 42
_TREE_MODELS = {"RandomForest"}  # models that must receive raw features


def _select_features(
    model_name: str,
    X_raw: np.ndarray,
    X_scaled: np.ndarray,
) -> np.ndarray:
    """Route raw vs scaled features based on model type."""
    return X_raw if model_name in _TREE_MODELS else X_scaled


# ── Baseline ──────────────────────────────────────────────────────────────────

def run_baseline(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_train_s: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_test_s: np.ndarray,
    n_runs: int = 3,
    class_weight: str | None = "balanced",
    n_classes: int | None = None,
    include_cnn: bool = True,
) -> pd.DataFrame:
    """
    Train all models n_runs times on the full training set.

    Returns a DataFrame with one row per (model, run), plus a summary
    row (mean ± std) per model appended at the end.
    """
    if n_classes is None:
        n_classes = int(np.max(y_train)) + 1
    print(f"\n  [baseline] class_weight={class_weight!r}, {n_runs} runs each, "
          f"n_classes={n_classes}")
    models_cfg = get_models(
        class_weight=class_weight,
        n_classes=n_classes,
        include_cnn=include_cnn,
    )
    records = []

    for model_name, model_template in models_cfg.items():
        X_tr = _select_features(model_name, X_train_raw, X_train_s)
        X_te = _select_features(model_name, X_test_raw,  X_test_s)
        run_buf = defaultdict(list)

        for run in range(n_runs):
            model = copy.deepcopy(model_template)
            t = train_model(model, X_tr, y_train)
            m = evaluate_model(model, X_te, y_test, n_classes=n_classes)

            run_buf["train_time_s"].append(t)
            for k in ("f1_macro", "f1_weighted", "precision_macro",
                      "recall_macro", "latency_ms", "throughput",
                      "auc_roc_macro", "model_size_bytes"):
                run_buf[k].append(m[k])

        # Aggregate across runs
        records.append({
            "model":              model_name,
            "class_weight":       str(class_weight),
            "train_time_mean":    np.mean(run_buf["train_time_s"]),
            "train_time_std":     np.std( run_buf["train_time_s"]),
            "f1_macro_mean":      np.mean(run_buf["f1_macro"]),
            "f1_macro_std":       np.std( run_buf["f1_macro"]),
            "f1_weighted_mean":   np.mean(run_buf["f1_weighted"]),
            "f1_weighted_std":    np.std( run_buf["f1_weighted"]),
            "precision_mean":     np.mean(run_buf["precision_macro"]),
            "recall_mean":        np.mean(run_buf["recall_macro"]),
            "auc_roc_mean":       np.nanmean(run_buf["auc_roc_macro"]),
            "latency_ms_mean":    np.mean(run_buf["latency_ms"]),
            "throughput_mean":    np.mean(run_buf["throughput"]),
            "model_size_mb":      np.mean(run_buf["model_size_bytes"]) / (1024 * 1024),
        })
        r = records[-1]
        print(
            f"    {model_name:15s}  "
            f"F1={r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}  "
            f"AUC={r['auc_roc_mean']:.4f}  "
            f"Train={r['train_time_mean']:.1f}s  "
            f"Lat={r['latency_ms_mean']:.3f}ms  "
            f"Size={r['model_size_mb']:.1f}MB"
        )

    return pd.DataFrame(records)


# ── Scalability ───────────────────────────────────────────────────────────────

def run_scalability(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_train_s: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_test_s: np.ndarray,
    fractions: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
    n_runs: int = 3,
    n_classes: int | None = None,
    include_cnn: bool = True,
) -> pd.DataFrame:
    """
    Train each model on subsets of increasing size, repeated n_runs times.

    Subset sampling is reproducible: each run uses a deterministic seed
    derived from RANDOM_STATE + run_index, so results are replicable.
    """
    if n_classes is None:
        n_classes = int(np.max(y_train)) + 1
    print("\n  [scalability] fractions:", fractions, f"| {n_runs} runs each")
    models_cfg = get_models(
        class_weight="balanced",
        n_classes=n_classes,
        include_cnn=include_cnn,
    )
    n_full = len(y_train)
    records = []

    for frac in fractions:
        n_sub = max(int(n_full * frac), 1)
        print(f"\n  ── {int(frac*100):3d}%  ({n_sub:,} training samples) ──")

        for model_name, model_template in models_cfg.items():
            X_tr_full = _select_features(model_name, X_train_raw, X_train_s)
            X_te      = _select_features(model_name, X_test_raw,  X_test_s)
            run_buf   = defaultdict(list)

            for run in range(n_runs):
                rng = np.random.default_rng(RANDOM_STATE + run)
                idx = rng.choice(n_full, size=n_sub, replace=False)

                model = copy.deepcopy(model_template)
                X_tr_sub = X_tr_full[idx]
                y_tr_sub = y_train[idx]

                t = train_model(model, X_tr_sub, y_tr_sub)
                m = evaluate_model(model, X_te, y_test, n_classes=n_classes)

                # Also score on the training subset itself — needed for a
                # textbook learning curve (train vs test F1 reveals the
                # bias-variance gap).  Cap the eval set at 50k samples so
                # large training fractions don't dominate runtime.
                cap = min(50_000, len(idx))
                eval_idx = idx[:cap]
                m_train = evaluate_model(
                    model, X_tr_full[eval_idx], y_train[eval_idx],
                    n_classes=n_classes,
                )

                run_buf["train_time_s"].append(t)
                run_buf["f1_macro_train"].append(m_train["f1_macro"])
                for k in ("f1_macro", "f1_weighted", "precision_macro",
                          "recall_macro", "latency_ms", "throughput",
                          "auc_roc_macro", "model_size_bytes"):
                    run_buf[k].append(m[k])

            records.append({
                "model":              model_name,
                "fraction":           frac,
                "n_train":            n_sub,
                "train_time_mean":    np.mean(run_buf["train_time_s"]),
                "train_time_std":     np.std( run_buf["train_time_s"]),
                "f1_macro_mean":      np.mean(run_buf["f1_macro"]),
                "f1_macro_std":       np.std( run_buf["f1_macro"]),
                "f1_macro_train_mean": np.mean(run_buf["f1_macro_train"]),
                "f1_macro_train_std":  np.std( run_buf["f1_macro_train"]),
                "f1_weighted_mean":   np.mean(run_buf["f1_weighted"]),
                "f1_weighted_std":    np.std( run_buf["f1_weighted"]),
                "precision_mean":     np.mean(run_buf["precision_macro"]),
                "recall_mean":        np.mean(run_buf["recall_macro"]),
                "auc_roc_mean":       np.nanmean(run_buf["auc_roc_macro"]),
                "latency_ms_mean":    np.mean(run_buf["latency_ms"]),
                "throughput_mean":    np.mean(run_buf["throughput"]),
                "model_size_mb":      np.mean(run_buf["model_size_bytes"]) / (1024 * 1024),
            })
            r = records[-1]
            print(
                f"    {model_name:15s}  "
                f"F1={r['f1_macro_mean']:.4f}±{r['f1_macro_std']:.4f}  "
                f"Train={r['train_time_mean']:.2f}s"
            )

    return pd.DataFrame(records)


# ── Class Weight Ablation ─────────────────────────────────────────────────────

def run_class_weight_study(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_train_s: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_test_s: np.ndarray,
    n_runs: int = 3,
    n_classes: int | None = None,
    include_cnn: bool = True,
) -> pd.DataFrame:
    """
    Compare class_weight=None vs class_weight="balanced" for all models.

    MLP does not accept class_weight, so its row is the same in both
    configurations; this is noted in the 'class_weight' column as 'N/A'.
    Including MLP in both conditions keeps the comparison table symmetric.
    The CNN DOES accept class_weight (via weighted CrossEntropyLoss), so
    its rows differ meaningfully across the two settings.
    """
    print("\n  [class_weight_study] None vs balanced  -  3 runs each")
    rows = []

    for cw in (None, "balanced"):
        df = run_baseline(
            X_train_raw, y_train, X_train_s,
            X_test_raw,  y_test,  X_test_s,
            n_runs=n_runs,
            class_weight=cw,
            n_classes=n_classes,
            include_cnn=include_cnn,
        )
        rows.append(df)

    return pd.concat(rows, ignore_index=True)
