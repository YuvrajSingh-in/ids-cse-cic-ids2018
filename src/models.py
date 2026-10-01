"""
models.py
─────────
Centralised model factory.

Each call to get_models() returns FRESH, unfitted instances.
This is intentional: experiments call this repeatedly so there
is zero risk of state leakage between runs.

class_weight parameter
──────────────────────
The IDS2018 dataset is severely imbalanced (Benign traffic vastly
outnumbers attack classes).  Two configurations are exposed:

  class_weight=None       → standard training; majority class dominates.
  class_weight="balanced" → samples are re-weighted inversely
                            proportional to class frequency, compensating
                            for imbalance without over/undersampling.

LinearSVC does not expose predict_proba natively; CalibratedClassifierCV
is intentionally avoided here to keep inference latency fair across models.

MLP configuration rationale
────────────────────────────
- (128, 64): two hidden layers capture non-linear feature interactions
  without excessive depth that would be unjustifiable on tabular data.
- early_stopping=True with validation_fraction=0.1 prevents over-fitting
  on large subsets; max_iter=100 is a hard cap.
- adam solver is default and appropriate for this dataset size.

CNN (deep learning model)
──────────────────────────
A 1D-CNN implemented in PyTorch is included as the deep learning method
required by the H9MLAI rubric.  Unlike sklearn's MLP, it:
  - uses a true deep-learning framework (PyTorch),
  - applies convolutional inductive bias over feature ordering,
  - supports class-weighted cross-entropy loss for imbalance,
  - exposes early stopping driven by a held-out validation split.
See src/cnn_model.py for architecture and training details.

n_classes for the CNN
──────────────────────
The CNN needs to know the number of output classes at construction time.
get_models() therefore takes an optional n_classes argument.  When None
(or include_cnn=False), the CNN is omitted from the returned dictionary
- useful for unit tests or quick traditional-only runs.
"""

import copy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

from src.cnn_model import IDS_CNN

RANDOM_STATE = 42


def get_models(
    class_weight: str | None = "balanced",
    n_classes: int | None = None,
    include_cnn: bool = True,
) -> dict:
    """
    Return a dictionary of fresh, unfitted classifier instances.

    Parameters
    ----------
    class_weight : {"balanced", None}
        Passed to models that support it.
    n_classes : int or None
        Number of output classes; required to instantiate the CNN.
        If None or include_cnn is False, the CNN is omitted.
    include_cnn : bool
        Convenience flag for skipping the CNN.

    Returns
    -------
    dict
        Keys are short model names; values are unfitted estimators.
    """
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "LinearSVM": LinearSVC(
            class_weight=class_weight,
            max_iter=2000,
            dual=False,
            random_state=RANDOM_STATE,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=RANDOM_STATE,
        ),
    }

    if include_cnn and n_classes is not None:
        models["CNN"] = IDS_CNN(
            n_classes=n_classes,
            class_weight=class_weight,
            epochs=50,
            batch_size=256,
            lr=1e-3,
            patience=5,
            random_state=RANDOM_STATE,
            verbose=False,
        )

    return models


def clone_model(model) -> object:
    """Return a deep copy of an unfitted model (resets any fitted state)."""
    return copy.deepcopy(model)
