"""
reproducibility.py
──────────────────
Reproducibility scaffolding:

1. save_run_metadata():     capture environment, seeds, dataset hash,
                            and experiment configuration to a JSON file
                            that accompanies the results for later audit.
2. persist_models():        serialise the best-performing models to disk
                            so the examiner can re-score without retraining.
3. persist_scaler():        save the fitted StandardScaler for identical
                            inference-side preprocessing.
4. load_persisted_model():  restore a saved model and run inference.
"""

import os
import sys
import json
import time
import platform
import hashlib
import pickle
from typing import Any


# ── Run metadata ──────────────────────────────────────────────────────────────

def save_run_metadata(
    output_path: str,
    *,
    random_state: int = 42,
    n_samples_total: int | None = None,
    n_features: int | None = None,
    n_classes: int | None = None,
    class_names: list[str] | None = None,
    class_weight: str = "balanced",
    extras: dict | None = None,
) -> None:
    """
    Write a JSON record describing the run.  Consumed by the report as the
    methodology appendix and by reviewers checking reproducibility.
    """
    meta: dict[str, Any] = {
        "timestamp_utc":     time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version":    sys.version.split()[0],
        "platform":          platform.platform(),
        "random_state":      random_state,
        "class_weight":      class_weight,
        "n_samples_total":   n_samples_total,
        "n_features":        n_features,
        "n_classes":         n_classes,
        "class_names":       class_names,
    }

    # Library versions (graceful degradation if absent)
    lib_versions = {}
    for lib in ("numpy", "pandas", "sklearn", "scipy", "matplotlib", "torch"):
        try:
            mod = __import__(lib)
            lib_versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            lib_versions[lib] = "not installed"
    meta["library_versions"] = lib_versions

    if extras:
        meta["extras"] = extras

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  [repro] Saved run metadata to {output_path}")


# ── Dataset hashing ───────────────────────────────────────────────────────────

def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """
    Compute SHA-256 of a file in streaming fashion (memory-safe for
    multi-GB inputs).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dataset_fingerprint(paths: list[str], output_path: str) -> dict[str, str]:
    """
    Compute SHA-256 for each dataset file and persist to JSON.  Allows the
    examiner to verify they have the identical CSVs referenced by the run.
    """
    fp = {os.path.basename(p): file_sha256(p) for p in paths if os.path.isfile(p)}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fp, f, indent=2)
    print(f"  [repro] Saved dataset fingerprints to {output_path}")
    return fp


# ── Model persistence ─────────────────────────────────────────────────────────

def persist_models(
    models: dict[str, object],
    output_dir: str,
) -> dict[str, str]:
    """
    Pickle each model to {output_dir}/<model_name>.pkl and return a map
    of model name → saved path.  Use load_persisted_model() to restore.

    Note on PyTorch models: IDS_CNN pickles cleanly because it holds only
    the module, history dict, and standard attributes.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for name, model in models.items():
        path = os.path.join(output_dir, f"{name}.pkl")
        try:
            with open(path, "wb") as f:
                pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
            paths[name] = path
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [repro] Saved {name} → {path}  ({size_mb:.2f} MB)")
        except Exception as exc:
            print(f"  [repro] Could not persist {name}: {exc}")
    return paths


def persist_scaler(scaler: object, output_dir: str) -> str:
    """Save the fitted StandardScaler so inference-side preprocessing is identical."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "scaler.pkl")
    with open(path, "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  [repro] Saved scaler → {path}")
    return path


def load_persisted_model(path: str) -> object:
    """Restore a model saved by persist_models()."""
    with open(path, "rb") as f:
        return pickle.load(f)
