"""
error_analysis.py
─────────────────
Post-hoc model diagnostics.

Functions
─────────
generate_confusion_matrix  → prints labelled confusion matrix table
print_classification_report → per-class precision/recall/F1
identify_hard_classes       → surfaces the most mis-classified classes
print_feature_importance    → top-N RF feature importances

Research value
──────────────
Confusion matrices reveal whether errors are symmetric (random
misclassification) or systematic (specific attack types consistently
confused with Benign or with each other).  In IDS research, false
negatives (attacks classified as Benign) are far more damaging than
false positives; this analysis makes that asymmetry visible.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def generate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
) -> pd.DataFrame:
    """
    Compute and print a labelled confusion matrix.

    Returns the matrix as a DataFrame (rows = true, cols = predicted).
    """
    labels = list(range(len(label_encoder.classes_)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    class_names = label_encoder.classes_
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

    print(f"\n  ── Confusion Matrix: {model_name} ──")
    # Truncate class names to 20 chars for display alignment
    short_names = [n[:20] for n in class_names]
    display_df = pd.DataFrame(
        cm,
        index=[f"T:{n}" for n in short_names],
        columns=[f"P:{n}" for n in short_names],
    )
    print(display_df.to_string())
    return cm_df


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
) -> None:
    """Print sklearn's per-class classification report."""
    report = classification_report(
        y_true, y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
    print(f"\n  ── Classification Report: {model_name} ──\n{report}")


def identify_hard_classes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
) -> dict:
    """
    Identify the classes with the highest error rates.

    Returns a dict with:
      - hardest_class     : class with lowest per-class accuracy
      - most_misclassified: class responsible for the most raw errors
    """
    classes = label_encoder.classes_
    n_classes = len(classes)
    error_rates = []
    error_counts = []

    for idx in range(n_classes):
        mask = y_true == idx
        if mask.sum() == 0:
            error_rates.append(0.0)
            error_counts.append(0)
            continue
        errors = (y_pred[mask] != idx).sum()
        error_rates.append(errors / mask.sum())
        error_counts.append(int(errors))

    hardest_idx        = int(np.argmax(error_rates))
    most_errors_idx    = int(np.argmax(error_counts))

    result = {
        "hardest_class":      classes[hardest_idx],
        "hardest_error_rate": error_rates[hardest_idx],
        "most_misclassified": classes[most_errors_idx],
        "most_errors_count":  error_counts[most_errors_idx],
    }

    print(f"\n  ── Hard-class Analysis: {model_name} ──")
    print(f"     Hardest class (highest error rate): "
          f"'{result['hardest_class']}'  "
          f"({result['hardest_error_rate']*100:.1f}% mis-classified)")
    print(f"     Most total errors: "
          f"'{result['most_misclassified']}'  "
          f"({result['most_errors_count']:,} samples wrong)")
    return result


def print_feature_importance(
    model: RandomForestClassifier,
    feature_names: list[str],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Display and return top-N feature importances from a fitted RF.

    Gini importances are used (default for sklearn RF).
    In IDS research, top features often correspond to flow-level
    statistics (packet length moments, inter-arrival time) that are
    exploited by specific attack signatures.
    """
    importances = model.feature_importances_
    ranked_idx = np.argsort(importances)[::-1][:top_n]
    rows = [
        {
            "rank":       rank + 1,
            "feature":    feature_names[i],
            "importance": importances[i],
        }
        for rank, i in enumerate(ranked_idx)
    ]
    df = pd.DataFrame(rows)
    print(f"\n  ── Top-{top_n} Feature Importances (RandomForest) ──")
    print(df.to_string(index=False))
    return df
