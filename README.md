# Intrusion Detection System on CSE-CIC-IDS2018

**MSc Artificial Intelligence — Machine Learning (H9MLAI) Research Project**
National College of Ireland, 2026

This repository contains the code, experimental pipeline, and reproducibility
artefacts for an empirical comparison of two traditional ML classifiers and
one deep-learning model on the CSE-CIC-IDS2018 network-flow dataset.

---

## At a glance

- **Dataset:** CSE-CIC-IDS2018 (3 selected days: Wed-14-02, Thu-15-02, Fri-02-03 ≈ 3.13M flows, 78 numeric features, 6 classes)
- **Models:**
    - Random Forest *(traditional ML, ensemble)*
    - Linear SVM *(traditional ML, linear)*
    - Multi-Layer Perceptron *(neural baseline)*
    - 1D Convolutional Neural Network in PyTorch *(deep learning)*
- **One-line run:** `./run_experiment.sh`

---

## Quick start

```bash
# 1. Place the three dataset CSVs in ./data/  (see Dataset section below)

# 2. One-shot reproducible run:
./run_experiment.sh

# Or manually:
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

All artefacts land in `./results/`:

```
results/
├── run_metadata.json              # environment, seeds, library versions
├── dataset_fingerprints.json      # SHA-256 of each input CSV
├── baseline_results.csv           # F1/AUC/latency/throughput/size per model
├── scalability_results.csv        # metrics at 25/50/75/100% training size
├── tuned_results.csv              # RandomizedSearchCV best configs
├── cv_per_fold.csv                # raw 5-fold CV scores
├── cv_summary.csv                 # CV mean ± 95% CI
├── feature_selection_results.csv  # MI/RFE/PCA comparison
├── feature_selections.json        # which features each method picked
├── sota_published_results.csv     # 12 curated published baselines
├── sota_comparison.csv            # our F1 vs published F1
├── eda/                           # descriptive stats + class dist + heatmaps
├── plots/                         # publication-ready PNGs
└── persisted_models/              # pickled models + scaler for re-scoring
```

---

## Dataset

CSE-CIC-IDS2018 is published by the Canadian Institute for Cybersecurity:
<https://www.unb.ca/cic/datasets/ids-2018.html>

Drop the following three CSVs into `data/`:

```
data/Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv  (~ 1.05M rows)
data/Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv   (~ 1.05M rows)
data/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv     (~ 1.05M rows)
```

The pipeline streams each file, drops timestamp columns, replaces +-Inf with
NaN, drops NaN rows, and concatenates. SHA-256 fingerprints of the loaded
files are written to `results/dataset_fingerprints.json` so an examiner can
verify they have the identical inputs.

---

## Repository layout

```
.
├── data/                          (place CSVs here, NOT in git)
├── src/
│   ├── data_loader.py             - CSV ingestion
│   ├── preprocessing.py           - clean/encode/split/scale
│   ├── eda.py                     - descriptive stats + plots
│   ├── feature_selection.py       - MI / RFE / PCA + comparison
│   ├── models.py                  - model factory (RF / SVM / MLP / CNN)
│   ├── cnn_model.py               - PyTorch 1D-CNN with AdamW + cosine LR
│   ├── evaluation.py              - metrics: F1, precision, recall, AUC,
│   │                                latency, throughput, model size
│   ├── experiments.py             - baseline / scalability / class-weight
│   ├── hyperparameter_tuning.py   - RandomizedSearchCV + manual CNN grid
│   ├── cross_validation.py        - 5-fold stratified CV with 95% CI
│   ├── parameter_sweep.py         - F1 vs CNN parameter count
│   ├── error_analysis.py          - confusion matrix + hard classes + FI
│   ├── plot_confusion.py          - per-model + grid confusion plots
│   ├── learning_curves.py         - train/test learning curve, CNN losses,
│   │                                complexity-vs-performance
│   ├── state_of_the_art.py        - 12 curated SOTA references + comparison
│   ├── reproducibility.py         - metadata, fingerprints, persistence
│   └── utils.py                   - console tables + scalability plots
├── main.py                        - end-to-end pipeline orchestrator
├── run_experiment.sh              - one-command reproducible runner
├── requirements.txt
└── README.md                      - this file
```

---

## What the pipeline does, step by step

`main.py` runs these stages in order. Each stage prints a banner and
saves its outputs.

| Stage | Module | Output |
|---|---|---|
| 1. Reproducibility setup | `reproducibility` | dataset SHA-256 + run metadata JSON |
| 2. Load + concatenate CSVs | `data_loader` | raw DataFrame |
| 3. Clean (drop timestamps, +-Inf, NaN), label-encode | `preprocessing` | feature matrix + labels |
| 4. 80/20 stratified train-test split | `preprocessing` | X_train, X_test, y_train, y_test |
| 5. StandardScaler fit on TRAIN only (no leakage) | `preprocessing` | scaled feature arrays + persisted scaler |
| 6. EDA (descriptive stats, class distribution, variance, correlation heatmap) | `eda` | CSVs + PNGs in `results/eda/` |
| 7. Baseline experiment (3 runs * 4 models, balanced class weights) | `experiments.run_baseline` | `baseline_results.csv` |
| 8. Feature selection study (MI-top20, RFE-top20, PCA-95%) | `feature_selection` | `feature_selection_results.csv` |
| 9. Scalability sweep (25/50/75/100% train size, 3 runs each) | `experiments.run_scalability` | `scalability_results.csv` (incl. train-set F1 for proper learning curves) |
| 10. Class-weight ablation (`None` vs `balanced`) | `experiments.run_class_weight_study` | `class_weight_comparison.csv` |
| 11. Hyperparameter tuning (RandomizedSearchCV, 3-fold inner CV) | `hyperparameter_tuning` | `tuned_results.csv` |
| 12. **5-fold stratified CV** with 95% confidence intervals | `cross_validation` | `cv_per_fold.csv` + `cv_summary.csv` |
| 13. CNN parameter-count sweep (5 capacities, F1 vs param count) | `parameter_sweep` | `cnn_param_sweep.csv` + plot |
| 14. Per-model error analysis (confusion matrix, classification report, hard-class summary, feature importance for RF, CNN loss-curve plot) | `error_analysis`, `plot_confusion`, `learning_curves` | console + `plots/` PNGs |
| 15. Persist all trained models + scaler | `reproducibility` | `persisted_models/*.pkl` |
| 16. SOTA comparison vs 12 curated published results | `state_of_the_art` | `sota_published_results.csv` + `sota_comparison.csv` |
| 17. Final plots: F1 vs size, training time vs size, model comparison, class-weight comparison, learning curves (train+test), complexity vs performance | `utils`, `learning_curves` | PNGs in `plots/` |

---

## Metrics reported

Every model produces, in every experiment:

| Metric | What it measures |
|---|---|
| **Precision-macro** | average per-class precision |
| **Recall-macro** | average per-class recall |
| **F1-macro** | harmonic mean of macro precision/recall (primary) |
| **F1-weighted** | support-weighted F1 (overall traffic mix) |
| **AUC-ROC (macro, OvR)** | one-vs-rest area under ROC, averaged across classes |
| **Training time (s)** | wall-clock fit duration |
| **Inference latency (ms/sample)** | mean per-sample predict time |
| **Throughput (samples/s)** | inverse of latency |
| **Model size (MB)** | pickle-serialised footprint, proxy for deployment cost |

For the CNN we additionally record:

- per-epoch train + validation cross-entropy loss
- per-epoch learning rate (`CosineAnnealingLR`)
- best-validation epoch (early stopping)
- L2 regularisation strength (`AdamW(weight_decay=1e-4)`)

---

## Reproducibility

This project takes reproducibility seriously because the rubric weights it
heavily. Every randomised operation is seeded with `random_state=42`:

- `numpy.random.default_rng(42)` for subsampling
- `sklearn` `random_state=42` for splits, RF, SVM, MLP, RandomizedSearchCV
- `torch.manual_seed(42)` and `torch.backends.cudnn.deterministic=True` for the CNN
- Stratified splits everywhere class proportions matter

Reproducibility artefacts:

- `results/run_metadata.json` - Python + library versions, platform, seeds, dataset shape
- `results/dataset_fingerprints.json` - SHA-256 of each input CSV
- `results/persisted_models/` - pickled models + fitted `StandardScaler`
  for examiner-side re-scoring without retraining

---

## System requirements

- Python >= 3.10
- ~ 8 GB RAM (full pipeline on full 3.13M rows)
- ~ 30 minutes wall-clock on a modern laptop CPU; faster with a GPU for the CNN

If memory is tight, edit `main.py` to subsample the dataset, or run the
sub-modules individually using their public entry-point functions.

---

## Acknowledgement of AI tool usage

In line with NCI policy on academic integrity: AI tools were used for
brainstorming, code-quality discussion, and explanation of library APIs.
Any specific use of generative AI is declared on the project cover sheet.
All experimental design decisions, methodology choices, and report writing
are the author's own.
