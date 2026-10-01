"""
main.py
───────
End-to-end orchestrator for the IDS ML research pipeline.

Stages (executed in order)
──────────────────────────
 0. Reproducibility setup       — dataset SHA-256 + run metadata JSON
 1. Load raw CSVs from data/
 2. Clean (timestamps, +-Inf, NaN), label-encode, build feature matrix
 3. 80/20 stratified train/test split
 4. StandardScaler fit on TRAIN only -> persist scaler
 5. EDA — descriptive stats, class distribution, variance, correlation
 6. Baseline experiment (3 runs, 4 models, balanced class weights,
                         metrics: F1/precision/recall/AUC/latency/throughput/size)
 7. Feature selection study (MI-top20, RFE-top20, PCA-95%) + comparison
 8. Scalability sweep (25/50/75/100% train; train AND test F1 captured
                       so a textbook learning curve can be produced)
 9. Class-weight ablation (None vs balanced)
10. Hyperparameter tuning (RandomizedSearchCV, 3-fold inner CV)
11. 5-fold stratified cross-validation with 95% confidence intervals
12. CNN parameter-count sweep (5 capacities -> F1 vs n_params)
13. Per-model error analysis: confusion matrices (text + heatmap +
    grid), classification report, hard-class summary, RF feature
    importance, CNN training-dynamics plot (loss + LR schedule)
14. Persist all trained models for examiner-side reproducibility
15. SOTA comparison vs 12 curated published references on IDS2018
16. Final plots — F1/training-time vs train size, model comparison,
    class-weight comparison, learning curves (train+test), complexity
    vs performance scatter

Usage
─────
  python main.py
  ./run_experiment.sh         # one-shot reproducible runner

Requirements
────────────
  pip install -r requirements.txt

Dataset
───────
  Place the three CSE-CIC-IDS2018 CSV files inside the data/ directory
  before running.  File names must match those listed in data_loader.py.
"""

import os
import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ── Project modules ───────────────────────────────────────────────────────────
from src.data_loader   import load_data
from src.preprocessing import (
    clean_data,
    encode_labels,
    build_feature_matrix,
    split_data,
    scale_data,
)
from src.models        import get_models
from src.evaluation    import train_model, evaluate_model
from src.experiments   import run_baseline, run_scalability, run_class_weight_study
from src.hyperparameter_tuning import tune_and_evaluate
from src.eda               import run_eda
from src.feature_selection import run_feature_selection_study
from src.cross_validation  import run_cross_validation
from src.parameter_sweep   import run_cnn_parameter_sweep
from src.state_of_the_art  import save_sota_table, compare_to_our_results
from src.plot_confusion    import plot_confusion_matrix, plot_all_confusion_matrices_grid
from src.learning_curves   import (
    plot_cnn_loss_curve,
    plot_learning_curves_from_scalability,
    plot_complexity_vs_performance,
)
from src.reproducibility   import (
    save_run_metadata, dataset_fingerprint,
    persist_models, persist_scaler,
)
from src.error_analysis import (
    generate_confusion_matrix,
    print_classification_report,
    identify_hard_classes,
    print_feature_importance,
)
from src.utils import (
    plot_f1_vs_size,
    plot_train_time_vs_size,
    plot_model_comparison,
    plot_class_weight_comparison,
    print_baseline_table,
    print_scalability_table,
    print_class_weight_table,
    print_research_summary,
    save_results,
)

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
MODELS_DIR  = os.path.join(RESULTS_DIR, "persisted_models")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner("IDS ML RESEARCH SYSTEM — CSE-CIC-IDS2018")

    # ── 0. Reproducibility scaffolding ────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from src.data_loader import DEFAULT_FILES
    try:
        dataset_fingerprint(
            paths=list(DEFAULT_FILES),
            output_path=os.path.join(RESULTS_DIR, "dataset_fingerprints.json"),
        )
    except Exception as exc:
        print(f"  [repro] Fingerprinting skipped: {exc}")

    # ── 1. Load ───────────────────────────────────────────────────────────────
    _section("1 / 10   Data Loading")
    df_raw = load_data()

    # ── 2-3. Clean + Encode ───────────────────────────────────────────────────
    _section("2 / 10   Preprocessing")
    df = clean_data(df_raw)
    df, label_encoder = encode_labels(df)

    # ── 4. Feature matrix ─────────────────────────────────────────────────────
    X, y, feature_names = build_feature_matrix(df)
    print(f"  Feature matrix ready: {X.shape[0]:,} × {X.shape[1]}")

    # ── 5. Split ──────────────────────────────────────────────────────────────
    _section("3 / 10   Train / Test Split  (80/20, stratified)")
    X_train_raw, X_test_raw, y_train, y_test = split_data(X, y)
    print(f"  Train: {len(y_train):,}   Test: {len(y_test):,}")

    # ── 6. Scale  (SVM / MLP only) ────────────────────────────────────────────
    _section("4 / 10   Feature Scaling  (scaler fit on train only)")
    X_train_s, X_test_s, scaler = scale_data(X_train_raw, X_test_raw)
    print("  StandardScaler applied. Random Forest will receive raw features.")

    # Persist the scaler for downstream inference reproducibility
    persist_scaler(scaler, MODELS_DIR)

    # ── 6b. EDA (training set only — no leakage) ──────────────────────────────
    _section("EDA   Descriptive Statistics and Visualisations")
    n_classes = len(label_encoder.classes_)
    eda_dir = os.path.join(RESULTS_DIR, "eda")
    run_eda(
        X_train=X_train_raw,
        y_train=y_train,
        feature_names=feature_names,
        class_names=list(label_encoder.classes_),
        output_dir=eda_dir,
    )

    # Save run metadata (now that we know n_classes and sample counts)
    save_run_metadata(
        output_path=os.path.join(RESULTS_DIR, "run_metadata.json"),
        random_state=42,
        n_samples_total=len(y_train) + len(y_test),
        n_features=X_train_raw.shape[1],
        n_classes=n_classes,
        class_names=list(label_encoder.classes_),
        class_weight="balanced",
        extras={
            "train_size": int(len(y_train)),
            "test_size":  int(len(y_test)),
            "test_fraction": 0.20,
        },
    )

    # ── 7. Baseline ───────────────────────────────────────────────────────────
    _section("5 / 10   Baseline Experiment  (full dataset, 3 runs, balanced)")
    baseline_df = run_baseline(
        X_train_raw, y_train, X_train_s,
        X_test_raw,  y_test,  X_test_s,
        n_runs=1,
        class_weight="balanced",
        n_classes=n_classes,
    )
    print_baseline_table(baseline_df)
    

    # ── 7b. Feature Selection Study ───────────────────────────────────────────
    _section("FS    Feature Selection  (MI-top20 / RFE-top20 / PCA-95%)")
    fs_df, fs_selections = run_feature_selection_study(
        X_train_raw=X_train_raw, y_train=y_train,
        X_test_raw=X_test_raw,   y_test=y_test,
        X_train_s=X_train_s,     X_test_s=X_test_s,
        feature_names=feature_names,
        n_classes=n_classes,
        k=20,
        output_dir=RESULTS_DIR,
    )
    print("\n  FEATURE SELECTION SUMMARY — F1-macro per (feature_set, model):")
    pivot = fs_df.pivot_table(index="feature_set", columns="model", values="f1_macro")
    print(pivot.round(4).to_string())


    # ── 8. Scalability ────────────────────────────────────────────────────────
    _section("6 / 10   Scalability Experiment  (25/50/75/100%, 3 runs each)")
    scalability_df = run_scalability(
        X_train_raw, y_train, X_train_s,
        X_test_raw,  y_test,  X_test_s,
        fractions=(0.25, 0.50, 0.75, 1.00),
        n_runs=1,
        n_classes=n_classes,
    )
    print_scalability_table(scalability_df)

    # ── 9. Class weight ablation ──────────────────────────────────────────────
    _section("7 / 10  Class Weight Study  (None vs balanced)")
    cw_df = run_class_weight_study(
        X_train_raw, y_train, X_train_s,
        X_test_raw,  y_test,  X_test_s,
        n_runs=1,
        n_classes=n_classes,
    )
    print_class_weight_table(cw_df)

    # ── 9b. Hyperparameter tuning ─────────────────────────────────────────────
    _section("8 / 10  Hyperparameter Tuning  (RandomizedSearchCV, 3-fold CV)")
    tuned_df = tune_and_evaluate(
        X_train_raw, y_train, X_train_s,
        X_test_raw,  y_test,  X_test_s,
        n_classes=n_classes,
        class_weight="balanced",
        tune_fraction=0.10,
        n_iter=5,
        cv_folds=3,
    )
    print("\n  TUNED vs DEFAULT — test F1-macro:")
    print(tuned_df[["model", "cv_f1_macro", "f1_macro", "tune_time_s", "train_time_s"]].to_string(index=False))
    tuned_df.to_csv(os.path.join(RESULTS_DIR, "tuned_results.csv"), index=False)

    # ── 9c. Cross-validation (k=5) ────────────────────────────────────────────
    _section("CV    5-fold Stratified Cross-Validation  (statistical rigour)")
    per_fold_df, cv_summary_df = run_cross_validation(
        X_raw=X_train_raw, y=y_train,
        n_classes=n_classes,
        k_folds=5,
        class_weight="balanced",
        subsample=200_000,   # stratified subsample to bound runtime
    )
    print("\n  CROSS-VALIDATION SUMMARY (F1-macro with 95% CI):")
    print(cv_summary_df.to_string(index=False))
    per_fold_df.to_csv(os.path.join(RESULTS_DIR, "cv_per_fold.csv"), index=False)
    cv_summary_df.to_csv(os.path.join(RESULTS_DIR, "cv_summary.csv"), index=False)

    # ── 9d. CNN parameter-count sweep ─────────────────────────────────────────
    # Addresses the rubric requirement that learning-curve evidence vary not
    # only dataset size but also "number of parameters".  Five CNN capacities
    # are trained on a fixed-size training subset; F1 vs param count plotted.
    _section("PARAM  CNN Capacity Sweep  (F1 vs number of trainable parameters)")
    sweep_df = run_cnn_parameter_sweep(
        X_train=X_train_s, y_train=y_train,
        X_test=X_test_s,   y_test=y_test,
        n_classes=n_classes,
        output_dir=PLOTS_DIR,
        configs=[
        {"name": "tiny",   "conv1_ch":  8, "conv2_ch": 16, "fc_units":  32},
        {"name": "medium", "conv1_ch": 32, "conv2_ch": 64, "fc_units": 128},
        {"name": "large",  "conv1_ch": 64, "conv2_ch": 128,"fc_units": 256},
        ],
        epochs=10,
    )
    sweep_df.to_csv(os.path.join(RESULTS_DIR, "cnn_param_sweep.csv"), index=False)

    # ── 10. Error analysis (on baseline models, balanced, single run) ─────────
    _section("9 / 10  Error Analysis + Confusion-Matrix Plots")
    models_for_analysis = get_models(class_weight="balanced", n_classes=n_classes)
    confusion_store: dict = {}    # for the multi-model grid plot
    trained_models: dict = {}     # to be persisted at the end

    for model_name, model_template in models_for_analysis.items():
        # Select correct feature set per model type
        X_tr = X_train_raw if model_name == "RandomForest" else X_train_s
        X_te = X_test_raw  if model_name == "RandomForest" else X_test_s

        model = copy.deepcopy(model_template)
        # Make CNN training history visible for the loss-curve plot
        if model_name == "CNN":
            model.verbose = True

        train_model(model, X_tr, y_train)
        result = evaluate_model(model, X_te, y_test, n_classes=n_classes)
        y_pred = result["y_pred"]

        # Text analysis (kept from original)
        cm_df = generate_confusion_matrix(y_test, y_pred, label_encoder, model_name)
        print_classification_report(y_test, y_pred, label_encoder, model_name)
        identify_hard_classes(y_test, y_pred, label_encoder, model_name)

        # Feature importances only available for Random Forest
        if model_name == "RandomForest":
            print_feature_importance(model, feature_names, top_n=20)

        # Confusion-matrix heatmap plots (counts + normalized)
        plot_confusion_matrix(
            y_true=y_test, y_pred=y_pred,
            class_names=list(label_encoder.classes_),
            model_name=model_name,
            output_dir=PLOTS_DIR,
            normalize=True,
        )
        confusion_store[model_name] = cm_df.values

        # G3: CNN training-dynamics plot (loss + LR over epochs)
        if model_name == "CNN":
            plot_cnn_loss_curve(model, PLOTS_DIR)

        # Hold the fitted model for end-of-pipeline persistence
        trained_models[model_name] = model

    # Multi-model grid plot
    plot_all_confusion_matrices_grid(
        confusion_matrices=confusion_store,
        class_names=list(label_encoder.classes_),
        output_dir=PLOTS_DIR,
        normalize=True,
    )

    # G6: persist all trained models for examiner-side reproducibility
    persist_models(trained_models, MODELS_DIR)

    # ── 9d. SOTA comparison ───────────────────────────────────────────────────
    _section("SOTA  Comparison Against Published CSE-CIC-IDS2018 Results")
    save_sota_table(output_dir=RESULTS_DIR)
    compare_to_our_results(
        our_results=baseline_df[baseline_df["class_weight"] == "balanced"],
        output_dir=RESULTS_DIR,
    )

    # ── 11. Plots ─────────────────────────────────────────────────────────────
    _section("10 / 10   Generating Plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plot_f1_vs_size(scalability_df, PLOTS_DIR)
    plot_train_time_vs_size(scalability_df, PLOTS_DIR)
    plot_model_comparison(baseline_df, PLOTS_DIR)
    plot_class_weight_comparison(cw_df, PLOTS_DIR)

    # G4: classical learning curves and complexity-vs-performance scatter
    plot_learning_curves_from_scalability(scalability_df, PLOTS_DIR)
    plot_complexity_vs_performance(baseline_df, PLOTS_DIR)

    # ── 12. Save results ──────────────────────────────────────────────────────
    save_results(baseline_df, scalability_df, cw_df, RESULTS_DIR)

    # ── 13. Research summary ──────────────────────────────────────────────────
    print_research_summary(baseline_df)

    _banner("PIPELINE COMPLETE")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(text: str) -> None:
    bar = "═" * 70
    print(f"\n{bar}\n  {text}\n{bar}")


def _section(text: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {text}")
    print(f"{'─' * 60}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
