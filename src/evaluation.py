"""
evaluation.py
─────────────
Training, inference, and metric computation.

Metrics reported
────────────────
- Precision / Recall / F1 (macro):  treat each class equally — critical
  when attack classes are rare and missed detections are costly.
- F1 (weighted):                    accounts for class support — useful
  for understanding overall system performance on real traffic mix.
- AUC-ROC (macro, one-vs-rest):     area under the ROC curve averaged
  across classes. Robust to imbalance; particularly informative when
  decision thresholds may need tuning at deployment.
- Training time (s):                wall-clock fit duration.
- Latency (ms/sample):              mean per-sample inference time.
- Throughput (samples/s):           inverse of latency; relevant for
                                    real-time deployment feasibility.
- Model size (bytes):               serialised object size, a proxy for
                                    memory footprint at deployment.

AUC-ROC for models without predict_proba
──────────────────────────────────────────
LinearSVC does not expose probabilistic outputs natively, so we fall
back to decision_function() and softmax-normalise the scores. This is
the standard technique: although not strictly calibrated, the ranking
produced by decision_function() is what AUC-ROC actually measures, so
the resulting AUC is meaningful.

Note on Random Forest and scaled data
──────────────────────────────────────
train_model and evaluate_model are agnostic to whether features are scaled.
The caller is responsible for passing the correct array (raw for RF,
scaled for SVM/MLP/CNN).  This separation keeps evaluation logic clean.
"""

import io
import pickle
import time
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
)
from scipy.special import softmax


def train_model(model, X_train: np.ndarray, y_train: np.ndarray) -> float:
    """
    Fit the model and return wall-clock training time in seconds.
    """
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    return time.perf_counter() - t0


def _get_probabilities(model, X: np.ndarray, n_classes: int) -> np.ndarray | None:
    """
    Return per-class scores suitable for AUC-ROC.

    Order of preference:
    1. predict_proba()          — calibrated probabilities
    2. decision_function()      — margin scores, softmax-normalised
    3. None                     — no scoring function available

    For multi-class LinearSVC, decision_function returns an (n, K) array
    of one-vs-rest margins, which we softmax to obtain pseudo-probabilities.
    The resulting AUC is equivalent to using the raw margins since AUC is
    invariant under monotonic transformations applied per class column.
    """
    # Prefer predict_proba (RF, MLP, CNN)
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)
            return np.asarray(proba)
        except Exception:
            pass

    # Fallback: decision_function (LinearSVC)
    if hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                # Binary case — pad to two columns
                scores = np.column_stack([-scores, scores])
            # Softmax-normalise for AUC compatibility
            return softmax(scores, axis=1)
        except Exception:
            pass

    return None


def _auc_roc_macro(y_true: np.ndarray, proba: np.ndarray | None, n_classes: int) -> float:
    """
    Compute macro-averaged one-vs-rest AUC-ROC.

    Returns NaN if probabilities are unavailable or if any class is missing
    from y_true (AUC is undefined in that case).
    """
    if proba is None:
        return float("nan")
    try:
        # Sanity: proba must be (n_samples, n_classes)
        if proba.ndim != 2 or proba.shape[1] != n_classes:
            return float("nan")
        if len(np.unique(y_true)) < n_classes:
            # Some class missing from the test fold — undefined for that class
            return float("nan")
        return float(roc_auc_score(
            y_true, proba,
            multi_class="ovr",
            average="macro",
        ))
    except Exception:
        return float("nan")


def _model_size_bytes(model) -> int:
    """
    Serialised model size in bytes (pickle protocol 5).

    Pickle size is a reasonable proxy for deployment footprint:
    - sklearn RF / SVM: reflects tree / coefficient memory
    - PyTorch CNN: reflects parameter + buffer state-dict size

    For a more precise deep-learning comparison you could sum
    torch parameter tensor .numel() * element_size(); pickle size
    over-estimates slightly due to Python bookkeeping, but the
    over-estimation is consistent across models.
    """
    try:
        buf = io.BytesIO()
        pickle.dump(model, buf, protocol=pickle.HIGHEST_PROTOCOL)
        return buf.tell()
    except Exception:
        return -1


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int | None = None,
) -> dict:
    """
    Run inference and compute all performance metrics.

    Parameters
    ----------
    model     : fitted estimator
    X_test    : feature matrix (must match the scale used during training)
    y_test    : true integer labels
    n_classes : number of classes (inferred if None)

    Returns
    -------
    dict with keys:
        precision_macro, recall_macro, f1_macro, f1_weighted, auc_roc_macro,
        latency_ms, throughput, model_size_bytes, y_pred
    """
    n = len(y_test)
    if n_classes is None:
        n_classes = int(np.max(y_test)) + 1

    # Inference timing
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    elapsed = time.perf_counter() - t0

    # Probabilistic output for AUC (not timed — inference latency is argmax-only)
    proba = _get_probabilities(model, X_test, n_classes)

    return {
        "precision_macro":  precision_score(y_test, y_pred, average="macro",    zero_division=0),
        "recall_macro":     recall_score(   y_test, y_pred, average="macro",    zero_division=0),
        "f1_macro":         f1_score(       y_test, y_pred, average="macro",    zero_division=0),
        "f1_weighted":      f1_score(       y_test, y_pred, average="weighted", zero_division=0),
        "auc_roc_macro":    _auc_roc_macro(y_test, proba, n_classes),
        "latency_ms":       (elapsed / n) * 1_000,
        "throughput":       n / elapsed,
        "model_size_bytes": _model_size_bytes(model),
        "y_pred":           y_pred,
    }
