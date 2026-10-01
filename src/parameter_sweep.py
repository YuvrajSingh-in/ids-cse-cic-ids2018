"""
parameter_sweep.py
──────────────────
Number-of-parameters experiment for the CNN.

Why
───
The H9MLAI rubric (criterion 3) explicitly mentions:

    "producing learning curve of your ML models on a varying sized
    data sets, data domains and number of parameters."

Most of the project varies dataset size (scalability) and hyperparameters
(tuning).  This module specifically varies the CNN's *parameter count* —
i.e. its capacity — and reports how F1-macro and overfitting gap respond.

Method
──────
Five CNN configurations of increasing capacity are trained on the same
training subset and evaluated on the same test set.  For each we record:
    - total trainable parameters,
    - test F1-macro,
    - train F1-macro (to surface the bias-variance gap),
    - training time,
    - inference latency.

The default configurations (kernel=3, stride=1) are kept fixed; only
channel widths and FC width vary.  This isolates the effect of capacity.
"""

import copy
import time
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from src.cnn_model import IDS_CNN, _CNN1D
from src.evaluation import train_model, evaluate_model

RANDOM_STATE = 42


def _count_parameters(model_pytorch: nn.Module) -> int:
    """Number of trainable parameters in a PyTorch model."""
    return sum(p.numel() for p in model_pytorch.parameters() if p.requires_grad)


class _CustomCapacityCNN(nn.Module):
    """
    Configurable-capacity 1D-CNN.  Defined at module scope (not inside
    a closure) so instances pickle cleanly — required for the
    model_size_bytes metric and for examiner-side reproducibility.
    """
    def __init__(self, n_features: int, n_classes: int,
                 conv1_ch: int, conv2_ch: int, fc_units: int,
                 dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, conv1_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv1_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(conv1_ch, conv2_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv2_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features)
            flat = self.features(dummy).view(1, -1).size(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fc_units, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def _build_cnn(
    n_features: int,
    n_classes: int,
    conv1_ch: int,
    conv2_ch: int,
    fc_units: int,
    dropout: float = 0.3,
) -> nn.Module:
    """
    Construct a custom-capacity CNN.  Thin wrapper around
    _CustomCapacityCNN so callers don't need to know about that class.
    """
    return _CustomCapacityCNN(
        n_features=n_features,
        n_classes=n_classes,
        conv1_ch=conv1_ch,
        conv2_ch=conv2_ch,
        fc_units=fc_units,
        dropout=dropout,
    )


class _CapacityCNN(IDS_CNN):
    """
    Subclass of IDS_CNN that builds a custom-capacity backbone instead of
    the default _CNN1D.  Inherits all of IDS_CNN's training, scoring,
    early-stopping, AdamW, and LR-scheduler behaviour.
    """
    def __init__(self, conv1_ch: int, conv2_ch: int, fc_units: int, **kwargs):
        super().__init__(**kwargs)
        self._conv1_ch = conv1_ch
        self._conv2_ch = conv2_ch
        self._fc_units = fc_units

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Replace the model construction step but otherwise reuse
        # the parent's training loop.  We monkey-patch by pre-building
        # the network and bypassing the default _CNN1D constructor.
        from src.cnn_model import _set_seed
        _set_seed(self.random_state)
        self.n_features_ = X.shape[1]

        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.10,
            random_state=self.random_state, stratify=y,
        )
        train_loader = self._make_loader(X_tr,  y_tr,  shuffle=True)
        val_loader   = self._make_loader(X_val, y_val, shuffle=False)

        self.model_ = _build_cnn(
            n_features=self.n_features_,
            n_classes=self.n_classes,
            conv1_ch=self._conv1_ch,
            conv2_ch=self._conv2_ch,
            fc_units=self._fc_units,
        ).to(self.device)

        # Re-use the parent's optimiser + scheduler logic by inlining it
        import torch.optim as optim
        loss_weight = self._compute_loss_weight(y_tr)
        criterion   = nn.CrossEntropyLoss(weight=loss_weight)
        optimizer = optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay,
        )
        if self.lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        else:
            scheduler = None

        best_val_loss = float("inf"); best_state = None; bad = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}
        for epoch in range(1, self.epochs + 1):
            tr_loss = self._train_one_epoch(train_loader, criterion, optimizer)
            v_loss  = self._eval_loss(val_loader, criterion)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(v_loss)
            history["lr"].append(optimizer.param_groups[0]["lr"])
            if v_loss < best_val_loss - 1e-4:
                best_val_loss = v_loss
                best_state = {k: t.clone() for k, t in self.model_.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break
            if scheduler is not None:
                scheduler.step()

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.history_ = history
        return self


# ── Public entry point ────────────────────────────────────────────────────────

def run_cnn_parameter_sweep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    output_dir: str,
    configs: list[dict] | None = None,
    epochs: int = 15,
) -> pd.DataFrame:
    """
    Train a series of CNNs of increasing capacity, return a DataFrame
    of (n_params, test_f1, train_f1, train_time, latency).

    A side-effect plot 'cnn_param_sweep.png' is written to output_dir
    showing F1 vs parameter count on a log-x scale.
    """
    if configs is None:
        # Five capacity points spanning ~3 orders of magnitude in params
        configs = [
            {"name": "tiny",   "conv1_ch":  8, "conv2_ch": 16, "fc_units":  32},
            {"name": "small",  "conv1_ch": 16, "conv2_ch": 32, "fc_units":  64},
            {"name": "medium", "conv1_ch": 32, "conv2_ch": 64, "fc_units": 128},
            {"name": "large",  "conv1_ch": 64, "conv2_ch": 128,"fc_units": 256},
            {"name": "xlarge", "conv1_ch": 96, "conv2_ch": 192,"fc_units": 384},
        ]

    os.makedirs(output_dir, exist_ok=True)
    rows = []
    print(f"\n  [param-sweep] Training {len(configs)} CNN sizes "
          f"(epochs={epochs} each)")

    for cfg in configs:
        cnn = _CapacityCNN(
            conv1_ch=cfg["conv1_ch"],
            conv2_ch=cfg["conv2_ch"],
            fc_units=cfg["fc_units"],
            n_classes=n_classes,
            class_weight="balanced",
            epochs=epochs,
            batch_size=256,
            lr=1e-3,
            weight_decay=1e-4,
            lr_schedule="cosine",
            patience=3,
            random_state=RANDOM_STATE,
            verbose=False,
        )

        # Build a parameter count BEFORE fit (so we have it even if fit fails)
        tmp = _build_cnn(
            n_features=X_train.shape[1],
            n_classes=n_classes,
            conv1_ch=cfg["conv1_ch"],
            conv2_ch=cfg["conv2_ch"],
            fc_units=cfg["fc_units"],
        )
        n_params = _count_parameters(tmp)
        del tmp

        t0 = time.time()
        train_time = train_model(cnn, X_train, y_train)
        test_result  = evaluate_model(cnn, X_test, y_test, n_classes=n_classes)
        # Train-set F1 (capped sample for fairness)
        cap = min(50_000, len(X_train))
        train_result = evaluate_model(cnn, X_train[:cap], y_train[:cap], n_classes=n_classes)
        elapsed = time.time() - t0

        rows.append({
            "name":         cfg["name"],
            "conv1_ch":     cfg["conv1_ch"],
            "conv2_ch":     cfg["conv2_ch"],
            "fc_units":     cfg["fc_units"],
            "n_params":     n_params,
            "test_f1":      test_result["f1_macro"],
            "train_f1":     train_result["f1_macro"],
            "test_auc":     test_result["auc_roc_macro"],
            "train_time_s": train_time,
            "latency_ms":   test_result["latency_ms"],
            "model_size_mb": test_result["model_size_bytes"] / (1024 * 1024),
        })
        print(f"    {cfg['name']:<7s}  params={n_params:>8,d}  "
              f"test_F1={test_result['f1_macro']:.4f}  "
              f"train_F1={train_result['f1_macro']:.4f}  "
              f"({elapsed:.1f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "cnn_param_sweep.csv"), index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(df["n_params"], df["test_f1"],
            marker="o", linewidth=2, color="#2166ac", label="Test F1")
    ax.plot(df["n_params"], df["train_f1"],
            marker="s", linewidth=2, color="#d6604d", label="Train F1",
            linestyle="--", alpha=0.8)
    for _, row in df.iterrows():
        ax.annotate(row["name"], (row["n_params"], row["test_f1"]),
                    xytext=(5, -12), textcoords="offset points", fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Number of trainable parameters (log)", fontsize=12)
    ax.set_ylabel("F1-Macro", fontsize=12)
    ax.set_title("CNN Capacity Sweep — F1 vs Parameter Count",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.4, which="both")
    ax.legend(fontsize=11, loc="lower right")
    fig.tight_layout()
    path = os.path.join(output_dir, "cnn_param_sweep.png")
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  [param-sweep] Saved {path}")
    return df
