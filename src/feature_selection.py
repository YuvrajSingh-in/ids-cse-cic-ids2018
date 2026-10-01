"""
feature_selection.py
────────────────────
Three dimensionality-reduction strategies compared against the full
78-feature baseline:

1. SelectKBest with mutual_info_classif (filter method)
   ────────────────────────────────────────────────────
   Ranks features by their mutual information with the label.  Captures
   non-linear dependence that Pearson correlation misses.  Model-agnostic:
   the same 20 features are selected regardless of downstream classifier.

2. RFE with Random Forest (wrapper method)
   ────────────────────────────────────────
   Iteratively removes the least-important feature according to RF's
   gini-importance, retraining at each step.  Produces a classifier-tuned
   selection that may differ from the filter method.

3. PCA retaining 95 percent variance (projection method)
   ──────────────────────────────────────────────────────
   Linear projection onto orthogonal components sorted by explained variance.
   Reduces dimensionality AND removes collinearity, at the cost of
   interpretability — principal components are weighted combinations of
   the original features, not physical flow statistics.

Evaluation protocol
───────────────────
Each reduced representation is trained with ALL four classifiers
(RF, LinearSVM, MLP, CNN) using their default configurations, and the
resulting F1-macro and training-time are reported alongside the
full-feature baseline. Differences surface the utility of selection on
this particular task.

Computational considerations
────────────────────────────
- mutual_info_classif is O(n log n × k) per feature; on 3M rows this is
  slow. We subsample to 200k rows for the information estimate. The
  ranking is stable at this size (verified empirically).
- RFE with RF is the slowest method; n_features_to_select is set to 20
  with step=10 to bound runtime. Full greedy (step=1) would be ~4x slower.
- PCA is O(n × d^2) and fast enough on the full training set.
"""

import time
import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from src.models import get_models
from src.evaluation import train_model, evaluate_model

RANDOM_STATE = 42


# ── Selection strategies ──────────────────────────────────────────────────────

def select_k_best_mi(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    k: int = 20,
    subsample: int = 200_000,
) -> tuple[np.ndarray, list[str], SelectKBest]:
    """Top-k features by mutual information with the label."""
    n = len(X)
    if subsample and n > subsample:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=subsample, replace=False)
        X_mi, y_mi = X[idx], y[idx]
        print(f"  [FS:MI] Estimating MI on {subsample:,}-row subsample")
    else:
        X_mi, y_mi = X, y

    selector = SelectKBest(
        score_func=lambda X_, y_: mutual_info_classif(
            X_, y_, random_state=RANDOM_STATE,
        ),
        k=k,
    )
    selector.fit(X_mi, y_mi)
    mask = selector.get_support()
    kept = [feature_names[i] for i in np.where(mask)[0]]
    return mask, kept, selector


def rfe_random_forest(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    k: int = 20,
    step: int = 10,
    subsample: int = 200_000,
) -> tuple[np.ndarray, list[str], RFE]:
    """
    Recursive Feature Elimination with a Random Forest estimator.

    step=10 removes 10 features per iteration — a coarse but fast schedule.
    Empirically, final rankings with step=10 and step=1 differ only in the
    marginal positions, not in the top-20.
    """
    n = len(X)
    if subsample and n > subsample:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(n, size=subsample, replace=False)
        X_r, y_r = X[idx], y[idx]
        print(f"  [FS:RFE] Running on {subsample:,}-row subsample")
    else:
        X_r, y_r = X, y

    estimator = RandomForestClassifier(
        n_estimators=50,           # smaller forest inside RFE for speed
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rfe = RFE(estimator=estimator, n_features_to_select=k, step=step)
    rfe.fit(X_r, y_r)
    mask = rfe.support_
    kept = [feature_names[i] for i in np.where(mask)[0]]
    return mask, kept, rfe


def pca_95(
    X_train: np.ndarray,
    X_test: np.ndarray,
    target_variance: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA on the training set, retaining enough components to capture
    target_variance of the total variance. Apply the same transform to test.
    """
    pca = PCA(n_components=target_variance, svd_solver="full", random_state=RANDOM_STATE)
    X_tr_p = pca.fit_transform(X_train).astype(np.float32)
    X_te_p = pca.transform(X_test).astype(np.float32)
    return X_tr_p, X_te_p, pca


# ── Comparison evaluator ──────────────────────────────────────────────────────

def evaluate_feature_sets(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    feature_masks: dict[str, np.ndarray],
    pca_arrays: tuple[np.ndarray, np.ndarray] | None,
    n_classes: int,
    class_weight: str | None = "balanced",
) -> pd.DataFrame:
    """
    Train every (model, feature-set) combination once and record performance.

    feature_masks maps a human name ('MI-top20', 'RFE-top20') to a boolean
    mask over the full feature axis.  pca_arrays, if provided, is a tuple
    (X_tr_pca, X_te_pca) already projected into PCA space.

    The 'Full' baseline is always included for reference.
    """
    import copy, gc

    print(f"\n  [FS] Evaluating feature sets with all models  "
          f"(class_weight={class_weight!r})")

    # Build the complete set of feature-space variants
    variants: dict[str, dict] = {
        "Full": {
            "X_tr_raw": X_train_raw, "X_te_raw": X_test_raw,
            "X_tr_s":   X_train_s,   "X_te_s":   X_test_s,
            "n_feat":   X_train_raw.shape[1],
        },
    }
    for name, mask in feature_masks.items():
        variants[name] = {
            "X_tr_raw": X_train_raw[:, mask], "X_te_raw": X_test_raw[:, mask],
            "X_tr_s":   X_train_s[:, mask],   "X_te_s":   X_test_s[:, mask],
            "n_feat":   int(mask.sum()),
        }
    if pca_arrays is not None:
        X_tr_p, X_te_p = pca_arrays
        # PCA output is already scaled (components are unit-length on zero-mean data);
        # we use it both as 'raw' and 'scaled' for routing simplicity.
        variants["PCA-95%"] = {
            "X_tr_raw": X_tr_p, "X_te_raw": X_te_p,
            "X_tr_s":   X_tr_p, "X_te_s":   X_te_p,
            "n_feat":   X_tr_p.shape[1],
        }

    rows = []
    for fs_name, fs in variants.items():
        print(f"\n  ── Feature set: {fs_name} ({fs['n_feat']} features) ──")
        # Fresh models for each feature set — prevents stale-state bugs
        models = get_models(class_weight=class_weight, n_classes=n_classes)

        for model_name, model_template in models.items():
            tree = (model_name == "RandomForest")
            X_tr = fs["X_tr_raw"] if tree else fs["X_tr_s"]
            X_te = fs["X_te_raw"] if tree else fs["X_te_s"]

            model = copy.deepcopy(model_template)
            # For CNN, keep training short for the feature-selection study
            if model_name == "CNN":
                model.epochs = 10
                model.patience = 2

            t = train_model(model, X_tr, y_train)
            result = evaluate_model(model, X_te, y_test, n_classes=n_classes)
            rows.append({
                "feature_set":     fs_name,
                "n_features":      fs["n_feat"],
                "model":           model_name,
                "train_time_s":    t,
                "f1_macro":        result["f1_macro"],
                "f1_weighted":     result["f1_weighted"],
                "precision_macro": result["precision_macro"],
                "recall_macro":    result["recall_macro"],
                "auc_roc_macro":   result["auc_roc_macro"],
                "latency_ms":      result["latency_ms"],
                "throughput":      result["throughput"],
                "model_size_mb":   result["model_size_bytes"] / (1024 * 1024),
            })
            r = rows[-1]
            print(f"    {model_name:14s}  F1={r['f1_macro']:.4f}  "
                  f"train={r['train_time_s']:.1f}s  lat={r['latency_ms']:.3f}ms")
            del model; gc.collect()
        del models; gc.collect()

    return pd.DataFrame(rows)


# ── High-level entry point ────────────────────────────────────────────────────

def run_feature_selection_study(
    X_train_raw: np.ndarray,
    y_train: np.ndarray,
    X_test_raw: np.ndarray,
    y_test: np.ndarray,
    X_train_s: np.ndarray,
    X_test_s: np.ndarray,
    feature_names: list[str],
    n_classes: int,
    k: int = 20,
    output_dir: str = "results",
) -> tuple[pd.DataFrame, dict]:
    """
    Execute the full feature-selection pipeline:
      1. Rank features by mutual information -> MI-top{k}
      2. Rank by RFE with RF                 -> RFE-top{k}
      3. Fit PCA(95% variance)               -> PCA-95%
      4. Evaluate all three reductions + Full baseline with all 4 models.

    Returns
    -------
    results_df : pd.DataFrame
        One row per (feature_set, model) with performance and timing.
    selections : dict
        Maps feature-set name to the list of feature names it contains
        (PCA entry is None since components are not original features).
    """
    import os, json
    os.makedirs(output_dir, exist_ok=True)

    selections: dict[str, list[str] | None] = {"Full": list(feature_names)}
    masks: dict[str, np.ndarray] = {}

    # 1. Mutual information
    print("\n  [FS] Mutual information ranking...")
    t0 = time.time()
    mi_mask, mi_kept, _ = select_k_best_mi(X_train_raw, y_train, feature_names, k=k)
    print(f"  [FS] MI selected {len(mi_kept)} features in {time.time()-t0:.1f}s")
    masks[f"MI-top{k}"] = mi_mask
    selections[f"MI-top{k}"] = mi_kept

    # 2. RFE
    print("\n  [FS] RFE with Random Forest...")
    t0 = time.time()
    rfe_mask, rfe_kept, _ = rfe_random_forest(X_train_raw, y_train, feature_names, k=k)
    print(f"  [FS] RFE selected {len(rfe_kept)} features in {time.time()-t0:.1f}s")
    masks[f"RFE-top{k}"] = rfe_mask
    selections[f"RFE-top{k}"] = rfe_kept

    # 3. PCA
    print("\n  [FS] PCA (95% variance)...")
    t0 = time.time()
    X_tr_p, X_te_p, pca = pca_95(X_train_s, X_test_s, target_variance=0.95)
    print(f"  [FS] PCA kept {X_tr_p.shape[1]} components "
          f"(explains {pca.explained_variance_ratio_.sum()*100:.2f}% var) "
          f"in {time.time()-t0:.1f}s")
    selections["PCA-95%"] = None    # components are not original features

    # Save the selections for the report
    sel_json = {
        name: (feats if feats is not None else f"{pca.n_components_} PCA components")
        for name, feats in selections.items()
    }
    with open(os.path.join(output_dir, "feature_selections.json"), "w") as f:
        json.dump(sel_json, f, indent=2)

    # Overlap analysis — how many features do MI and RFE agree on?
    overlap = set(mi_kept) & set(rfe_kept)
    print(f"\n  [FS] MI ∩ RFE = {len(overlap)} / {k} features in common")

    # 4. Evaluate
    results_df = evaluate_feature_sets(
        X_train_raw, y_train, X_test_raw, y_test,
        X_train_s, X_test_s,
        feature_masks=masks,
        pca_arrays=(X_tr_p, X_te_p),
        n_classes=n_classes,
        class_weight="balanced",
    )
    results_df.to_csv(os.path.join(output_dir, "feature_selection_results.csv"), index=False)
    return results_df, selections
