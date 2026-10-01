"""
state_of_the_art.py
───────────────────
Published state-of-the-art results on CSE-CIC-IDS2018.

This module provides a curated table of results reported in the IDS
literature for the same dataset used in this project. The final report
compares this project's measured performance against these published
numbers as the rubric-mandated "baseline" evaluation.

Notes on comparability
──────────────────────
Direct numerical comparison between studies is NEVER perfect. Published
papers differ in:
  - which day files / attack classes they use,
  - whether they report macro or weighted F1,
  - whether they rebalance with SMOTE / undersampling before evaluation,
  - whether they evaluate binary or multi-class classification,
  - whether they tune hyperparameters and how aggressively.

Each reference row carries enough metadata (experimental setup column)
for a reader to interpret the comparison fairly. Citations use the
author-year format; the full bibliographic entries belong in the report's
references section (IEEE style).
"""

import os
import pandas as pd


# ── Curated SOTA table ────────────────────────────────────────────────────────
# Each row is a single published result on CSE-CIC-IDS2018.
# "model" is the best-performing algorithm in that paper.
# f1_macro is the reported macro-F1 when available; otherwise f1_weighted
#   or overall F1 is noted in the 'f1_type' column.
# Scores are taken as reported in the cited paper — we do NOT normalise or
# re-compute them. See the paper's setup column for methodology.

SOTA_RESULTS: list[dict] = [
    {
        "citation":     "Farhan & Jasim (2022)",
        "venue":        "Indonesian Journal of Electrical Eng. & CS, 26(2)",
        "model":        "LSTM",
        "accuracy":     0.991,
        "f1_score":     0.991,
        "f1_type":      "overall",
        "task":         "multi-class (7 classes)",
        "setup":        "Oversampling + feature scaling; reported as accuracy ~= F1",
    },
    {
        "citation":     "Hagar & Gawali (2022)",
        "venue":        "Computational Intelligence & Neuroscience",
        "model":        "Apache Spark (RF + PCA)",
        "accuracy":     1.000,
        "f1_score":     1.000,
        "f1_type":      "weighted",
        "task":         "multi-class (15 classes)",
        "setup":        "RF feature selection (19/84); SMOTE + undersampling; Spark MLlib",
    },
    {
        "citation":     "Hagar & Gawali (2022)",
        "venue":        "Computational Intelligence & Neuroscience",
        "model":        "CNN",
        "accuracy":     0.996,
        "f1_score":     0.996,
        "f1_type":      "weighted",
        "task":         "multi-class (15 classes)",
        "setup":        "Same as above row; CNN with 200 hidden nodes, Adam, softmax",
    },
    {
        "citation":     "Hagar & Gawali (2022)",
        "venue":        "Computational Intelligence & Neuroscience",
        "model":        "LSTM",
        "accuracy":     0.995,
        "f1_score":     0.995,
        "f1_type":      "weighted",
        "task":         "multi-class (15 classes)",
        "setup":        "Same as above row; LSTM with 200 hidden nodes, Adam, softmax",
    },
    {
        "citation":     "Mohamed et al. (2024)",
        "venue":        "IJECE, 11(3)",
        "model":        "Gradient Boosting",
        "accuracy":     None,
        "f1_score":     0.95,
        "f1_type":      "macro",
        "task":         "multi-class (6 classes)",
        "setup":        "RF regressor feature selection; undersampled to a balanced subset",
    },
    {
        "citation":     "Mohamed et al. (2024)",
        "venue":        "IJECE, 11(3)",
        "model":        "Random Forest",
        "accuracy":     None,
        "f1_score":     0.93,
        "f1_type":      "macro",
        "task":         "multi-class (6 classes)",
        "setup":        "Same as above row",
    },
    {
        "citation":     "Mohamed et al. (2024)",
        "venue":        "IJECE, 11(3)",
        "model":        "MLP",
        "accuracy":     None,
        "f1_score":     0.78,
        "f1_type":      "macro",
        "task":         "multi-class (6 classes)",
        "setup":        "Same as above row",
    },
    {
        "citation":     "Lawal et al. (2025)",
        "venue":        "NIPES J. Science & Tech. Research, 7 Spec.",
        "model":        "XGBoost",
        "accuracy":     0.96,
        "f1_score":     0.96,
        "f1_type":      "weighted",
        "task":         "multi-class",
        "setup":        "Undersampling for balance; data cleaning + scaling",
    },
    {
        "citation":     "Lawal et al. (2025)",
        "venue":        "NIPES J. Science & Tech. Research, 7 Spec.",
        "model":        "Random Forest",
        "accuracy":     None,
        "f1_score":     0.95,
        "f1_type":      "weighted",
        "task":         "multi-class",
        "setup":        "Same as above row",
    },
    {
        "citation":     "Cosar et al. (2024)",
        "venue":        "AI Theory & Applications, 4(2)",
        "model":        "XGBoost / LightGBM / CatBoost",
        "accuracy":     0.98,
        "f1_score":     None,
        "f1_type":      None,
        "task":         "multi-class",
        "setup":        "Reported accuracy ~0.98 across boosted tree family; AUC=1.00",
    },
    {
        "citation":     "Songma et al. (2023)",
        "venue":        "Computers, 12(12) - MDPI",
        "model":        "XGBoost",
        "accuracy":     0.99995,
        "f1_score":     0.9789,
        "f1_type":      "macro",
        "task":         "multi-class",
        "setup":        "Z-score + min-max normalisation; 10-fold CV; best in study",
    },
    {
        "citation":     "Adib et al. (2025)",
        "venue":        "JUKTISI, 4(3)",
        "model":        "CNN",
        "accuracy":     0.99,
        "f1_score":     0.99,
        "f1_type":      "weighted",
        "task":         "binary (DDoS vs benign)",
        "setup":        "Flow-feature based DDoS detection; CNN best DL model",
    },
]


# ── Build / load table ────────────────────────────────────────────────────────

def build_sota_table() -> pd.DataFrame:
    """Return the curated SOTA table as a DataFrame."""
    return pd.DataFrame(SOTA_RESULTS)


def save_sota_table(output_dir: str) -> pd.DataFrame:
    """Save the SOTA table to CSV and return it."""
    os.makedirs(output_dir, exist_ok=True)
    df = build_sota_table()
    path = os.path.join(output_dir, "sota_published_results.csv")
    df.to_csv(path, index=False)
    print(f"  [sota] Saved {path}  ({len(df)} published results)")
    return df


# ── Comparison to our own results ─────────────────────────────────────────────

def compare_to_our_results(
    our_results: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """
    Produce a side-by-side comparison of our project's per-model best F1
    against representative published SOTA rows.

    our_results must contain columns 'model' and 'f1_macro' and may contain
    additional columns; they will be ignored.

    The output table has one row per our-model with the closest-comparable
    SOTA reference listed alongside. "Closest" here is a pragmatic mapping:
    same model family, macro-F1 where available.
    """
    sota = build_sota_table()

    # Hand-picked mapping: our model -> (SOTA citation, SOTA model family filter)
    # We prefer rows with f1_type == 'macro' since that matches our reporting.
    mapping = {
        "RandomForest": ("Mohamed et al. (2024)", "Random Forest"),
        "LinearSVM":    None,    # no matching SOTA row in our curated set
        "MLP":          ("Mohamed et al. (2024)", "MLP"),
        "CNN":          ("Hagar & Gawali (2022)", "CNN"),
    }

    rows = []
    for _, our_row in our_results.iterrows():
        model_name = our_row["model"]
        # Accept either 'f1_macro' (single-run) or 'f1_macro_mean' (aggregated)
        if "f1_macro" in our_row.index:
            ours_f1 = float(our_row["f1_macro"])
        elif "f1_macro_mean" in our_row.index:
            ours_f1 = float(our_row["f1_macro_mean"])
        else:
            raise KeyError(
                "our_results must contain either 'f1_macro' or 'f1_macro_mean'"
            )

        ref = mapping.get(model_name)
        if ref is None:
            rows.append({
                "our_model":       model_name,
                "our_f1_macro":    ours_f1,
                "sota_citation":   "no direct match in curated set",
                "sota_model":      "-",
                "sota_f1":         None,
                "sota_f1_type":    "-",
                "delta":           None,
                "comment":         "LinearSVM rarely reported as a SOTA baseline on IDS2018 "
                                   "(tree ensembles and deep learning dominate).",
            })
            continue

        citation, sota_model = ref
        sub = sota[(sota["citation"] == citation) & (sota["model"] == sota_model)]
        if sub.empty:
            continue
        sota_row = sub.iloc[0]
        sota_f1 = sota_row["f1_score"]
        delta = ours_f1 - float(sota_f1) if sota_f1 is not None else None

        rows.append({
            "our_model":       model_name,
            "our_f1_macro":    round(ours_f1, 4),
            "sota_citation":   citation,
            "sota_model":      sota_model,
            "sota_f1":         sota_f1,
            "sota_f1_type":    sota_row["f1_type"],
            "delta":           round(delta, 4) if delta is not None else None,
            "comment":         sota_row["setup"],
        })

    df = pd.DataFrame(rows)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "sota_comparison.csv")
    df.to_csv(path, index=False)
    print(f"\n  [sota] Comparison saved to {path}")
    print("\n  OUR RESULTS vs PUBLISHED SOTA (macro F1 where reported):")
    print(df.to_string(index=False))
    return df
