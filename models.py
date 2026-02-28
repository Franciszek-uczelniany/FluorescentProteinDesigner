"""Model definitions, training, saving, loading, and inference.

Supports Ridge regression, MLP, and ensemble models. Each trained model is saved
as a self-contained artifact with everything needed for inference (PCA, scalers, etc.).
"""

import json
import os
from collections import OrderedDict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split


# ── Model presets ─────────────────────────────────────────────────────────────
PRESETS = {
    "ridge": {"pooling": "augmented", "pca": 512},
    "mlp":   {"pooling": "mean",      "pca": 256},
}


# ── MLP architecture ─────────────────────────────────────────────────────────
def _build_mlp(input_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 1),
    )


# ── Training ──────────────────────────────────────────────────────────────────
def train_ridge(X_train: np.ndarray, y_train: np.ndarray,
                pca_n: int = 512, embedding: str = "esm2",
                pooling: str = "augmented") -> dict:
    """Train Ridge regression with optional PCA. Returns artifact dict."""
    pca = None
    if pca_n is not None and pca_n < X_train.shape[1]:
        pca = PCA(n_components=pca_n, random_state=42)
        X_train = pca.fit_transform(X_train)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {pca_n} components, {explained:.1%} variance explained")

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    return {
        "model_type": "ridge",
        "embedding": embedding,
        "pooling": pooling,
        "pca": pca,
        "pca_n": pca_n,
        "model": model,
    }


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              pca_n: int = 256, embedding: str = "esm2",
              pooling: str = "mean") -> dict:
    """Train MLP with standardization and early stopping. Returns artifact dict."""
    pca = None
    if pca_n is not None and pca_n < X_train.shape[1]:
        pca = PCA(n_components=pca_n, random_state=42)
        X_train = pca.fit_transform(X_train)
        explained = pca.explained_variance_ratio_.sum()
        print(f"  PCA: {pca_n} components, {explained:.1%} variance explained")

    input_dim = X_train.shape[1]

    # Validation split for early stopping
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )
    print(f"  Splits: {len(X_tr)} train / {len(X_val)} val")

    # Standardize features
    X_mean = X_tr.mean(axis=0)
    X_std = X_tr.std(axis=0) + 1e-8
    X_tr = (X_tr - X_mean) / X_std
    X_val = (X_val - X_mean) / X_std

    # Standardize targets
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std()) + 1e-8
    y_tr_z = (y_tr - y_mean) / y_std
    y_val_z = (y_val - y_mean) / y_std

    # Tensors
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr_z, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val_z, dtype=torch.float32, device=device).unsqueeze(1)

    # Model
    model = _build_mlp(input_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  MLP parameters: {total_params:,}")

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, 501):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_val_t), y_val_t).item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"    epoch {epoch:3d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 30:
                print(f"    Early stopping at epoch {epoch} (best val_loss={best_val_loss:.4f})")
                break

    model.load_state_dict(best_state)

    return {
        "model_type": "mlp",
        "embedding": embedding,
        "pooling": pooling,
        "pca": pca,
        "pca_n": pca_n,
        "input_dim": input_dim,
        "X_mean": X_mean,
        "X_std": X_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "state_dict": OrderedDict({k: v.cpu() for k, v in model.state_dict().items()}),
        "architecture": [input_dim, 256, 64, 1],
    }


# ── Prediction ────────────────────────────────────────────────────────────────
def predict(artifact: dict, X: np.ndarray) -> np.ndarray:
    """Run inference using a loaded artifact. Handles PCA + scaling internally."""
    if artifact["model_type"] == "ensemble":
        return _predict_ensemble(artifact, X)

    # Apply PCA if present
    if artifact.get("pca") is not None:
        X = artifact["pca"].transform(X)

    if artifact["model_type"] == "ridge":
        return artifact["model"].predict(X)

    elif artifact["model_type"] == "mlp":
        # Standardize
        X_scaled = (X - artifact["X_mean"]) / artifact["X_std"]

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = _build_mlp(artifact["input_dim"]).to(device)
        model.load_state_dict(artifact["state_dict"])
        model.eval()

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pred = model(X_t).cpu().numpy().squeeze()

        return y_pred * artifact["y_std"] + artifact["y_mean"]

    raise ValueError(f"Unknown model type: {artifact['model_type']}")


def _predict_ensemble(artifact: dict, X: np.ndarray) -> np.ndarray:
    """Average predictions from ensemble components.

    X should be a dict with {"mean": ndarray, "augmented": ndarray} for ensemble,
    or we use the appropriate pooling from each component.
    """
    preds = []
    weights = artifact["weights"]
    for comp, w in zip(artifact["components"], weights):
        # Each component needs its own pooling type of embeddings
        if isinstance(X, dict):
            X_comp = X[comp["pooling"]]
        else:
            X_comp = X
        preds.append(predict(comp, X_comp) * w)
    return sum(preds)


# ── Save / Load ───────────────────────────────────────────────────────────────
def save_artifact(artifact: dict, path: Path) -> None:
    """Save a trained model artifact to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if artifact["model_type"] == "ensemble":
        # Ensemble is a JSON pointer to component files (relative to ensemble dir)
        comp_rel_paths = []
        for c in artifact["component_paths"]:
            c = Path(c)
            try:
                comp_rel_paths.append(str(c.relative_to(path.parent)))
            except ValueError:
                comp_rel_paths.append(str(os.path.relpath(c, path.parent)))
        with open(path, "w") as f:
            json.dump({
                "model_type": "ensemble",
                "embedding": artifact["embedding"],
                "components": comp_rel_paths,
                "weights": artifact["weights"],
                "metrics": artifact.get("metrics"),
            }, f, indent=2)
    elif artifact["model_type"] == "ridge":
        joblib.dump(artifact, path)
    elif artifact["model_type"] == "mlp":
        torch.save(artifact, path)

    print(f"  Saved {path}")


def load_artifact(path: Path) -> dict:
    """Load a model artifact from disk."""
    path = Path(path)

    if path.suffix == ".json":
        with open(path) as f:
            meta = json.load(f)
        # Load component artifacts
        components = []
        for comp_name in meta["components"]:
            comp_path = path.parent / comp_name
            components.append(load_artifact(comp_path))
        return {
            "model_type": "ensemble",
            "embedding": meta["embedding"],
            "components": components,
            "weights": meta["weights"],
            "metrics": meta.get("metrics"),
        }

    elif path.suffix == ".joblib":
        return joblib.load(path)

    elif path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)

    raise ValueError(f"Unknown artifact format: {path.suffix}")


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R2."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(metrics: dict, label: str = "") -> None:
    """Pretty-print evaluation metrics."""
    prefix = f"{label}: " if label else ""
    print(f"  {prefix}MAE={metrics['MAE']:.2f} nm  RMSE={metrics['RMSE']:.2f} nm  R2={metrics['R2']:.4f}")
