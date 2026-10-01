"""
cnn_model.py
────────────
1D Convolutional Neural Network for tabular network-flow intrusion detection.

Why a CNN on tabular data?
──────────────────────────
While CNNs are most associated with images, 1D-CNNs have been successfully
applied to network flow features in IDS literature (e.g., Vinayakumar et al.,
2019; Kim et al., 2020). The convolution operation captures local patterns
across adjacent statistical features (e.g., consecutive packet-length moments,
inter-arrival time statistics) that fully-connected networks must learn from
scratch. This inductive bias often improves sample efficiency on flow data.

Architecture (deliberately compact for CPU training)
─────────────────────────────────────────────────────
    Input:        (batch, 1, n_features)         e.g. (256, 1, 79)
    Conv1d(1→32, k=3, pad=1) + ReLU + BN         → (256, 32, 79)
    MaxPool1d(2)                                 → (256, 32, 39)
    Conv1d(32→64, k=3, pad=1) + ReLU + BN        → (256, 64, 39)
    MaxPool1d(2)                                 → (256, 64, 19)
    Flatten                                      → (256, 1216)
    Dropout(0.3)
    Linear(1216 → 128) + ReLU
    Dropout(0.3)
    Linear(128 → n_classes)                      → logits

Training details
────────────────
- Optimiser:   Adam (lr=1e-3)
- Loss:        CrossEntropy with optional class weights
- Batch size:  256
- Early stop:  patience=5 epochs on validation loss
- Max epochs:  50
- Validation:  10% holdout from training data, stratified

Sklearn-compatible interface
────────────────────────────
The IDS_CNN class exposes .fit(X, y) and .predict(X), so it slots into the
existing train_model() and evaluate_model() helpers without modification.
This keeps the experimental pipeline clean and unified.

Reproducibility
───────────────
Seeds are set on Python, NumPy, and PyTorch (CPU + CUDA). Deterministic
algorithms are enabled where supported. Note: full bitwise reproducibility
on GPU requires CUDA-specific flags that may slow training; we accept this
trade-off and document it.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# ── Reproducibility ──────────────────────────────────────────────────────────

def _set_seed(seed: int) -> None:
    """Seed all RNGs that affect training."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Network definition ──────────────────────────────────────────────────────

class _CNN1D(nn.Module):
    """
    1D-CNN classifier for tabular flow features.

    Two conv blocks (with BatchNorm and MaxPool) followed by a small MLP head.
    BatchNorm stabilises training under varying batch composition; Dropout
    on the dense layers reduces over-fitting on the dominant Benign class.
    """

    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        # Compute flattened size dynamically (handles odd n_features)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_features)
            flat_size = self.features(dummy).view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(flat_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


# ── Sklearn-compatible wrapper ──────────────────────────────────────────────

class IDS_CNN:
    """
    Sklearn-style wrapper around _CNN1D.

    Exposes .fit(X, y), .predict(X), and .predict_proba(X) so the model can
    be used interchangeably with sklearn estimators in the experiment loops.

    Parameters
    ----------
    n_classes      : number of output classes
    class_weight   : 'balanced' or None — controls imbalance handling
    epochs         : maximum training epochs (early stopping may stop sooner)
    batch_size     : mini-batch size
    lr             : Adam initial learning rate
    weight_decay   : L2 penalty strength (passed to AdamW).  Default 1e-4
                     provides mild regularisation consistent with IDS
                     literature on CSE-CIC-IDS2018.
    lr_schedule    : 'none' | 'cosine' | 'step'.  'cosine' uses
                     CosineAnnealingLR over `epochs`; 'step' halves the
                     learning rate every 5 epochs.
    patience       : epochs without val-loss improvement before stopping
    device         : 'cpu' or 'cuda' — auto-detected if None
    random_state   : seed for full reproducibility
    verbose        : print per-epoch loss if True
    """

    def __init__(
        self,
        n_classes: int,
        class_weight: str | None = "balanced",
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_schedule: str = "cosine",
        patience: int = 5,
        device: str | None = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_classes    = n_classes
        self.class_weight = class_weight
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.lr           = lr
        self.weight_decay = weight_decay
        self.lr_schedule  = lr_schedule
        self.patience     = patience
        self.random_state = random_state
        self.verbose      = verbose

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Set after fit
        self.model_: _CNN1D | None = None
        self.n_features_: int | None = None
        self.history_: dict | None = None

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray, y: np.ndarray) -> "IDS_CNN":
        """Train the CNN with early stopping on a stratified validation split."""
        _set_seed(self.random_state)

        self.n_features_ = X.shape[1]

        # Stratified validation split (10%) for early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y,
            test_size=0.10,
            random_state=self.random_state,
            stratify=y,
        )

        train_loader = self._make_loader(X_tr,  y_tr,  shuffle=True)
        val_loader   = self._make_loader(X_val, y_val, shuffle=False)

        # Build network and move to device
        self.model_ = _CNN1D(
            n_features=self.n_features_,
            n_classes=self.n_classes,
        ).to(self.device)

        # Loss with optional class weights
        loss_weight = self._compute_loss_weight(y_tr)
        criterion = nn.CrossEntropyLoss(weight=loss_weight)

        # AdamW applies decoupled weight-decay (true L2) — preferred over
        # Adam's coupled variant.  See Loshchilov & Hutter (2019).
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Learning-rate scheduler
        if self.lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs,
            )
        elif self.lr_schedule == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=5, gamma=0.5,
            )
        else:
            scheduler = None

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state    = None
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(1, self.epochs + 1):
            tr_loss  = self._train_one_epoch(train_loader, criterion, optimizer)
            val_loss = self._eval_loss(val_loader, criterion)
            current_lr = optimizer.param_groups[0]["lr"]

            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)
            history["lr"].append(current_lr)

            if self.verbose:
                print(f"    [CNN] epoch {epoch:02d}  "
                      f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
                      f"lr={current_lr:.2e}")

            # Early stopping check
            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                best_state    = {k: v.clone() for k, v in self.model_.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    if self.verbose:
                        print(f"    [CNN] early stop at epoch {epoch}")
                    break

            # Step the LR scheduler AFTER the epoch
            if scheduler is not None:
                scheduler.step()

        # Restore best weights
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.history_ = history
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class predictions as an int array."""
        logits = self._forward_eval(X)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return softmax probabilities — useful for ROC/threshold analysis."""
        logits = self._forward_eval(X)
        return torch.softmax(logits, dim=1).cpu().numpy()

    # ── Internals ───────────────────────────────────────────────────────────

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        """Wrap arrays in a DataLoader with the correct shape (N, 1, F)."""
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)   # (N, 1, F)
        y_t = torch.from_numpy(y.astype(np.int64))
        ds  = TensorDataset(X_t, y_t)
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,        # 0 keeps reproducibility & avoids fork issues
            drop_last=False,
        )

    def _compute_loss_weight(self, y: np.ndarray) -> torch.Tensor | None:
        """Convert 'balanced' setting into a per-class weight tensor."""
        if self.class_weight != "balanced":
            return None
        classes = np.arange(self.n_classes)
        present = np.unique(y)
        # compute_class_weight requires every class to appear; pad missing
        weights = np.ones(self.n_classes, dtype=np.float32)
        if len(present) == self.n_classes:
            cw = compute_class_weight("balanced", classes=classes, y=y)
            weights = cw.astype(np.float32)
        else:
            cw = compute_class_weight("balanced", classes=present, y=y)
            for cls, w in zip(present, cw):
                weights[cls] = w
        return torch.from_numpy(weights).to(self.device)

    def _train_one_epoch(self, loader, criterion, optimizer) -> float:
        self.model_.train()
        running = 0.0
        n_seen  = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            logits = self.model_(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n_seen  += xb.size(0)
        return running / max(n_seen, 1)

    @torch.no_grad()
    def _eval_loss(self, loader, criterion) -> float:
        self.model_.eval()
        running = 0.0
        n_seen  = 0
        for xb, yb in loader:
            xb = xb.to(self.device, non_blocking=True)
            yb = yb.to(self.device, non_blocking=True)
            logits = self.model_(xb)
            loss   = criterion(logits, yb)
            running += loss.item() * xb.size(0)
            n_seen  += xb.size(0)
        return running / max(n_seen, 1)

    @torch.no_grad()
    def _forward_eval(self, X: np.ndarray) -> torch.Tensor:
        """Forward pass in eval mode, batched to control memory."""
        if self.model_ is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        self.model_.eval()
        X_t = torch.from_numpy(X.astype(np.float32)).unsqueeze(1)
        out_chunks = []
        for i in range(0, len(X_t), self.batch_size):
            xb = X_t[i:i + self.batch_size].to(self.device)
            out_chunks.append(self.model_(xb))
        return torch.cat(out_chunks, dim=0)
