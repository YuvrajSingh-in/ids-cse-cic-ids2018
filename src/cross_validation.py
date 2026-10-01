"""
cross_validation.py
───────────────────
Stratified k-fold cross-validation for all four classifiers.

Purpose
───────
The baseline experiment uses a single 80/20 train/test split. This is
standard practice but gives a point estimate of performance with no
indication of its variance. The H9MLAI rubric emphasises statistical
rigour, so we add a 5-fold stratified CV step that reports:

  - per-fold F1-macro,
  - mean F1-macro across folds,
  - standard deviation across folds,
  - 95% confidence interval (t-distribution, df = k - 1).

Design choices
──────────────
- StratifiedKFold preserves class proportions in each fold — essential
  on this severely imbalanced dataset (DoS-Slowloris at 0.35%).
- k = 5 is the common benchmark. k = 10 would halve each fold's hold-out
  and roughly double total runtime without substantially tightening CIs.
- For each model we use a SUBSAMPLE of the training data for CV, if
  requested. On a 640k-row training set, 5-fold CV with four models and
  a deep CNN would take hours; a 100k-row stratified subsample gives a
  fair variance estimate at a fraction of the cost. The subsample size
  is exposed as a parameter.
- No hyperparameter tuning inside the CV loop; this is pure performance
  estimation, not model selection. (Tuning uses its own internal CV
  inside RandomizedSearchCV; see hyperparameter_tuning.py.)

Output
──────
Returns a DataFrame with one row per (model, fold) and a companion summary
DataFrame with one row per model (mean, std, 95% CI).
"""

import copy
import time
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.models import get_models
from src.evaluation import train_model, evaluate_model

RANDOM_STATE = 42
_TREE_MODELS = {"RandomForest"}


def run_cross_validation(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    k_folds: int = 5,
    class_weight: str | None = "balanced",
    subsample: int | None = None,
    include_cnn: bool = True,
    models_to_run: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run k-fold stratified CV on all requested models.

    Parameters
    ----------
    X_raw        : raw (unscaled) feature matrix
    y            : integer label vector
    n_classes    : number of output classes (needed for CNN)
    k_folds      : number of CV folds (default 5)
    class_weight : passed through to models
    subsample    : optional stratified subsample size (for runtime)
    include_cnn  : whether to include the CNN in the CV sweep
    models_to_run: optional tuple to restrict which models run

    Returns
    -------
    per_fold_df : DataFrame with columns [model, fold, f1_macro, ...]
    summary_df  : DataFrame with columns [model, mean, std, ci95_low, ci95_high]
    """
    # Optional subsampling, stratified, with a deterministic seed
    if subsample and len(y) > subsample:
        rng = np.random.default_rng(RANDOM_STATE)
        keep = []
        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            n_keep = max(int(len(cls_idx) * subsample / len(y)), min(500, len(cls_idx)))
            n_keep = min(n_keep, len(cls_idx))
            keep.append(rng.choice(cls_idx, size=n_keep, replace=False))
        keep_idx = np.sort(np.concatenate(keep))
        X_raw = X_raw[keep_idx]
        y = y[keep_idx]
        print(f"  [cv] Subsampled to {len(y):,} rows for CV")

    print(f"  [cv] Running {k_folds}-fold stratified CV on {len(y):,} samples")

    # Model list
    all_models = get_models(
        class_weight=class_weight,
        n_classes=n_classes,
        include_cnn=include_cnn,
    )
    if models_to_run:
        all_models = {k: v for k, v in all_models.items() if k in models_to_run}

    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=RANDOM_STATE)
    per_fold_rows = []

    for model_name, model_template in all_models.items():
        print(f"\n  ── {model_name}: {k_folds}-fold CV ──")
        fold_scores = []
        t_model = time.time()

        for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_raw, y), start=1):
            X_tr_raw, X_te_raw = X_raw[tr_idx], X_raw[te_idx]
            y_tr,     y_te     = y[tr_idx],    y[te_idx]

            # Scale within-fold (fit on fold's train only — no leakage)
            scaler = StandardScaler().fit(X_tr_raw)
            X_tr_s = scaler.transform(X_tr_raw).astype(np.float32)
            X_te_s = scaler.transform(X_te_raw).astype(np.float32)

            # Feature routing
            tree = (model_name in _TREE_MODELS)
            X_tr = X_tr_raw if tree else X_tr_s
            X_te = X_te_raw if tree else X_te_s

            # Fresh model per fold — no state leak
            model = copy.deepcopy(model_template)
            if model_name == "CNN":
                # Keep CNN training bounded in CV for wall-clock budget
                model.epochs = 10
                model.patience = 2

            t0 = time.time()
            train_time = train_model(model, X_tr, y_tr)
            result = evaluate_model(model, X_te, y_te, n_classes=n_classes)
            elapsed = time.time() - t0

            per_fold_rows.append({
                "model":           model_name,
                "fold":            fold_idx,
                "train_time_s":    train_time,
                "fold_time_s":     elapsed,
                "f1_macro":        result["f1_macro"],
                "f1_weighted":     result["f1_weighted"],
                "precision_macro": result["precision_macro"],
                "recall_macro":    result["recall_macro"],
                "auc_roc_macro":   result["auc_roc_macro"],
                "model_size_mb":   result["model_size_bytes"] / (1024 * 1024),
            })
            fold_scores.append(result["f1_macro"])
            print(f"    fold {fold_idx}/{k_folds}  "
                  f"f1_macro={result['f1_macro']:.4f}  "
                  f"({elapsed:.1f}s)")

            # Free memory between folds
            del model, scaler, X_tr_s, X_te_s
            import gc; gc.collect()

        mean = float(np.mean(fold_scores))
        std  = float(np.std(fold_scores, ddof=1))
        # 95% CI via t-distribution (appropriate for small k)
        if len(fold_scores) > 1:
            tval = stats.t.ppf(0.975, df=len(fold_scores) - 1)
            margin = tval * std / np.sqrt(len(fold_scores))
        else:
            margin = 0.0
        ci_low, ci_high = mean - margin, mean + margin
        print(f"    mean f1_macro = {mean:.4f} ± {std:.4f}  "
              f"(95% CI: [{ci_low:.4f}, {ci_high:.4f}])  "
              f"[{time.time() - t_model:.1f}s total]")

    per_fold_df = pd.DataFrame(per_fold_rows)

    # Summary per model
    summary_rows = []
    for model_name, grp in per_fold_df.groupby("model"):
        scores = grp["f1_macro"].values
        mean = float(scores.mean())
        std  = float(scores.std(ddof=1))
        tval = stats.t.ppf(0.975, df=len(scores) - 1)
        margin = tval * std / np.sqrt(len(scores))
        summary_rows.append({
            "model":         model_name,
            "n_folds":       len(scores),
            "mean_f1_macro": round(mean, 4),
            "std_f1_macro":  round(std, 4),
            "ci95_low":      round(mean - margin, 4),
            "ci95_high":     round(mean + margin, 4),
            "mean_train_s":  round(grp["train_time_s"].mean(), 2),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "mean_f1_macro", ascending=False,
    ).reset_index(drop=True)

    return per_fold_df, summary_df
