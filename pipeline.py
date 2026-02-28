#!/usr/bin/env python
"""Training pipeline for fluorescent protein emission wavelength prediction.

Usage:
    python pipeline.py                                  # full pipeline (cross-embedding ensemble)
    python pipeline.py run --model cross_ensemble       # explicit cross-embedding ensemble
    python pipeline.py run --embedding esm2 --model ensemble
    python pipeline.py download
    python pipeline.py embed --embedding esmc600m
    python pipeline.py train --model ridge --pca 512
    python pipeline.py evaluate --model cross_ensemble
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import data
import embeddings
import models

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

_USE_PRESET = object()  # sentinel: use model preset PCA value


# ── Helpers ───────────────────────────────────────────────────────────────────
def _artifact_path(embedding: str, model_name: str) -> Path:
    """Return the expected artifact path for a given model config."""
    if model_name == "cross_ensemble":
        return ARTIFACTS_DIR / "cross" / "ensemble.json"

    preset = models.PRESETS.get(model_name, {})
    pooling = preset.get("pooling", "mean")
    pca_n = preset.get("pca", 256)

    if model_name == "ridge":
        suffix = f"_pca{pca_n}" if pca_n is not None else "_full"
        return ARTIFACTS_DIR / embedding / f"ridge_{pooling}{suffix}.joblib"
    elif model_name == "mlp":
        suffix = f"_pca{pca_n}" if pca_n is not None else "_full"
        return ARTIFACTS_DIR / embedding / f"mlp_{pooling}{suffix}.pt"
    elif model_name == "ensemble":
        return ARTIFACTS_DIR / embedding / "ensemble.json"
    raise ValueError(f"Unknown model: {model_name}")


def _get_embeddings_for_model(model_name: str, emb_data: dict) -> np.ndarray:
    """Select the right pooling type for a model."""
    pooling = models.PRESETS.get(model_name, {}).get("pooling", "mean")
    return emb_data[pooling]


def _export_predictions(
    model_label: str,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    names_test: np.ndarray,
    metrics: dict,
    artifact_dir: Path,
) -> Path:
    """Save per-sample ground truth vs predicted to a JSON file alongside the artifact."""
    df = data.get_dataset()
    name_to_seq = dict(zip(df["name"], df["sequence"]))

    records = []
    for i in range(len(y_test)):
        name = str(names_test[i])
        records.append({
            "name": name,
            "sequence": name_to_seq.get(name, ""),
            "ground_truth_emission_nm": round(float(y_test[i]), 1),
            "predicted_emission_nm": round(float(y_pred[i]), 1),
            "absolute_error_nm": round(float(abs(y_test[i] - y_pred[i])), 1),
        })
    records.sort(key=lambda r: r["name"])

    output = {
        "model": model_label,
        "n_test_samples": len(records),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "predictions": records,
    }

    path = artifact_dir / f"test_predictions_{model_label}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved test predictions to {path}")
    return path


# ── Subcommands ───────────────────────────────────────────────────────────────
def cmd_download(args):
    data.download()
    df = data.get_dataset()
    print(f"\n{len(df)} proteins with sequence + em_max")
    print(f"em_max range: {df['em_max'].min():.0f} - {df['em_max'].max():.0f} nm")


def cmd_embed(args):
    emb = embeddings.get_embeddings(args.embedding)
    print(f"\nEmbeddings ready ({args.embedding}):")
    print(f"  Mean:      {emb['mean'].shape}")
    print(f"  Augmented: {emb['augmented'].shape}")
    print(f"  Proteins:  {len(emb['names'])}")


def cmd_train(args):
    model_name = args.model

    if model_name == "cross_ensemble":
        _train_cross_ensemble(args.force)
        return

    embedding_name = args.embedding
    emb = embeddings.get_embeddings(embedding_name)

    if model_name == "ensemble":
        _train_ensemble(emb, embedding_name, args.force)
    else:
        pca = args.pca if args.pca is not None else _USE_PRESET
        _train_single(model_name, emb, embedding_name, pca, args.force)


def _train_single(model_name: str, emb: dict, embedding_name: str,
                   pca_override: int | None | object = _USE_PRESET,
                   force: bool = False) -> Path:
    """Train a single model (ridge or mlp). Returns artifact path.

    pca_override: _USE_PRESET = use model preset, None = no PCA, int = specific PCA.
    """
    preset = models.PRESETS[model_name]
    pooling = preset["pooling"]
    pca_n = preset["pca"] if pca_override is _USE_PRESET else pca_override

    suffix = f"_pca{pca_n}" if pca_n is not None else "_full"
    path = ARTIFACTS_DIR / embedding_name / f"{model_name}_{pooling}{suffix}"
    path = path.with_suffix(".joblib" if model_name == "ridge" else ".pt")

    if path.exists() and not force:
        print(f"Artifact exists: {path} (use --force to retrain)")
        return path

    X = emb[pooling]
    y = emb["em_max"]
    names = emb["names"]

    X_train, X_test, y_train, y_test, _, names_test = data.get_split(X, y, names)

    print(f"\nTraining {model_name} ({embedding_name}, {pooling}, PCA {pca_n})...")
    print(f"  Train: {len(X_train)}  Test: {len(X_test)}")

    if model_name == "ridge":
        artifact = models.train_ridge(X_train, y_train, pca_n=pca_n,
                                       embedding=embedding_name, pooling=pooling)
    else:
        artifact = models.train_mlp(X_train, y_train, pca_n=pca_n,
                                     embedding=embedding_name, pooling=pooling)

    # Evaluate on test set
    y_pred = models.predict(artifact, X_test)
    metrics = models.evaluate(y_test, y_pred)
    artifact["metrics"] = metrics
    models.print_metrics(metrics, model_name)

    models.save_artifact(artifact, path)
    _export_predictions(model_name, y_test, y_pred, names_test,
                        metrics, ARTIFACTS_DIR / embedding_name)
    return path


def _train_ensemble(emb: dict, embedding_name: str, force: bool):
    """Train ensemble = Ridge + MLP, then save ensemble.json."""
    ensemble_path = ARTIFACTS_DIR / embedding_name / "ensemble.json"

    if ensemble_path.exists() and not force:
        print(f"Artifact exists: {ensemble_path} (use --force to retrain)")
        return

    # Train components (they'll skip if already exist and not forced)
    ridge_path = _train_single("ridge", emb, embedding_name, None, force)
    mlp_path = _train_single("mlp", emb, embedding_name, None, force)

    # Load components and evaluate ensemble on test set
    ridge_artifact = models.load_artifact(ridge_path)
    mlp_artifact = models.load_artifact(mlp_path)

    # Get test split for each pooling type
    y = emb["em_max"]
    names = emb["names"]

    X_aug = emb["augmented"]
    _, X_aug_test, y_train, y_test, _, names_test = data.get_split(X_aug, y, names)

    X_mean = emb["mean"]
    _, X_mean_test, _, _, _, _ = data.get_split(X_mean, y, names)

    y_pred_ridge = models.predict(ridge_artifact, X_aug_test)
    y_pred_mlp = models.predict(mlp_artifact, X_mean_test)
    y_pred_ensemble = (y_pred_ridge + y_pred_mlp) / 2

    metrics = models.evaluate(y_test, y_pred_ensemble)

    ensemble_artifact = {
        "model_type": "ensemble",
        "embedding": embedding_name,
        "component_paths": [ridge_path, mlp_path],
        "weights": [0.5, 0.5],
        "metrics": metrics,
    }

    models.save_artifact(ensemble_artifact, ensemble_path)
    _export_predictions("ensemble", y_test, y_pred_ensemble, names_test,
                        metrics, ARTIFACTS_DIR / embedding_name)
    print(f"\nEnsemble results:")
    models.print_metrics(metrics, "Ensemble")

    # Print comparison table
    ridge_m = ridge_artifact.get("metrics", {})
    mlp_m = mlp_artifact.get("metrics", {})
    print(f"\n{'Model':<20s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}")
    print("-" * 48)
    for name, m in [("Ridge", ridge_m), ("MLP", mlp_m), ("Ensemble", metrics)]:
        if m:
            print(f"{name:<20s} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['R2']:>8.4f}")


def _train_cross_ensemble(force: bool):
    """Train cross-embedding ensemble: ESM-2 augmented Ridge + ESM-C mean MLP."""
    ensemble_path = ARTIFACTS_DIR / "cross" / "ensemble.json"

    if ensemble_path.exists() and not force:
        print(f"Artifact exists: {ensemble_path} (use --force to retrain)")
        return

    # Load both embedding models
    esm2_emb = embeddings.get_embeddings("esm2")
    esmc_emb = embeddings.get_embeddings("esmc600m")

    # Train Ridge on ESM-2 augmented (no PCA — full dimensionality)
    ridge_path = _train_single("ridge", esm2_emb, "esm2", pca_override=None, force=force)

    # Train MLP on ESM-C mean (no PCA — full dimensionality)
    mlp_path = _train_single("mlp", esmc_emb, "esmc600m", pca_override=None, force=force)

    # Load components and evaluate ensemble on test set
    ridge_artifact = models.load_artifact(ridge_path)
    mlp_artifact = models.load_artifact(mlp_path)

    y = esm2_emb["em_max"]
    names = esm2_emb["names"]

    # Ridge uses ESM-2 augmented
    _, X_ridge_test, _, y_test, _, names_test = data.get_split(
        esm2_emb["augmented"], y, names
    )
    # MLP uses ESM-C mean
    _, X_mlp_test, _, _, _, _ = data.get_split(
        esmc_emb["mean"], y, names
    )

    y_pred_ridge = models.predict(ridge_artifact, X_ridge_test)
    y_pred_mlp = models.predict(mlp_artifact, X_mlp_test)
    y_pred_ensemble = (y_pred_ridge + y_pred_mlp) / 2

    metrics = models.evaluate(y_test, y_pred_ensemble)

    ensemble_artifact = {
        "model_type": "ensemble",
        "embedding": "cross (esm2 + esmc600m)",
        "component_paths": [ridge_path, mlp_path],
        "weights": [0.5, 0.5],
        "metrics": metrics,
    }

    models.save_artifact(ensemble_artifact, ensemble_path)
    _export_predictions("cross_ensemble", y_test, y_pred_ensemble, names_test,
                        metrics, ARTIFACTS_DIR / "cross")

    print(f"\nCross-embedding ensemble results:")
    models.print_metrics(metrics, "Cross Ensemble")

    ridge_m = ridge_artifact.get("metrics", {})
    mlp_m = mlp_artifact.get("metrics", {})
    print(f"\n{'Model':<35s} {'MAE':>8s} {'RMSE':>8s} {'R2':>8s}")
    print("-" * 63)
    for name, m in [("Ridge (ESM-2 augmented)", ridge_m),
                    ("MLP (ESM-C mean)", mlp_m),
                    ("Cross Ensemble", metrics)]:
        if m:
            print(f"{name:<35s} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} {m['R2']:>8.4f}")


def cmd_evaluate(args):
    path = _artifact_path(args.embedding, args.model)
    if not path.exists():
        print(f"Artifact not found: {path}")
        print(f"Run: python pipeline.py train --model {args.model}")
        sys.exit(1)

    artifact = models.load_artifact(path)

    if args.model == "cross_ensemble":
        esm2_emb = embeddings.get_embeddings("esm2")
        esmc_emb = embeddings.get_embeddings("esmc600m")
        y = esm2_emb["em_max"]
        names = esm2_emb["names"]

        _, X_ridge_test, _, y_test, _, _ = data.get_split(
            esm2_emb["augmented"], y, names
        )
        _, X_mlp_test, _, _, _, _ = data.get_split(
            esmc_emb["mean"], y, names
        )
        # Components: [0]=ridge(esm2 aug), [1]=mlp(esmc mean)
        y_pred_ridge = models.predict(artifact["components"][0], X_ridge_test)
        y_pred_mlp = models.predict(artifact["components"][1], X_mlp_test)
        y_pred = (y_pred_ridge + y_pred_mlp) / 2
    elif args.model == "ensemble":
        emb = embeddings.get_embeddings(args.embedding)
        y = emb["em_max"]
        names = emb["names"]
        _, X_aug_test, _, y_test, _, _ = data.get_split(emb["augmented"], y, names)
        _, X_mean_test, _, _, _, _ = data.get_split(emb["mean"], y, names)
        X_test = {"mean": X_mean_test, "augmented": X_aug_test}
        y_pred = models.predict(artifact, X_test)
    else:
        emb = embeddings.get_embeddings(args.embedding)
        y = emb["em_max"]
        names = emb["names"]
        pooling = models.PRESETS[args.model]["pooling"]
        X = emb[pooling]
        _, X_test, _, y_test, _, _ = data.get_split(X, y, names)
        y_pred = models.predict(artifact, X_test)

    metrics = models.evaluate(y_test, y_pred)
    models.print_metrics(metrics, args.model)


def cmd_run(args):
    """Full pipeline: download -> embed -> train -> evaluate."""
    is_cross = args.model == "cross_ensemble"
    label = "cross_ensemble (esm2 + esmc600m)" if is_cross else f"{args.embedding} / {args.model}"

    print("=" * 60)
    print(f"Full pipeline: {label}")
    print("=" * 60)

    # Step 1: Download
    print("\n[1/4] Dataset")
    df = data.get_dataset()
    print(f"  {len(df)} proteins")

    # Step 2: Embed
    if is_cross:
        print(f"\n[2/4] Embeddings (esm2 + esmc600m)")
        esm2_emb = embeddings.get_embeddings("esm2")
        print(f"  ESM-2 — Mean: {esm2_emb['mean'].shape}, Augmented: {esm2_emb['augmented'].shape}")
        esmc_emb = embeddings.get_embeddings("esmc600m")
        print(f"  ESM-C — Mean: {esmc_emb['mean'].shape}, Augmented: {esmc_emb['augmented'].shape}")
    else:
        print(f"\n[2/4] Embeddings ({args.embedding})")
        emb = embeddings.get_embeddings(args.embedding)
        print(f"  Mean: {emb['mean'].shape}, Augmented: {emb['augmented'].shape}")

    # Step 3: Train
    print(f"\n[3/4] Training ({args.model})")
    if is_cross:
        _train_cross_ensemble(args.force)
    elif args.model == "ensemble":
        _train_ensemble(emb, args.embedding, args.force)
    else:
        pca = args.pca if args.pca is not None else _USE_PRESET
        _train_single(args.model, emb, args.embedding, pca, args.force)

    # Step 4: Evaluate
    print(f"\n[4/4] Evaluation")
    cmd_evaluate(args)

    print("\nDone!")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fluorescent protein emission wavelength prediction pipeline"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Default: run
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--embedding", choices=["esm2", "esmc600m", "esmc6b"], default="esm2")
    common.add_argument("--model", choices=["ridge", "mlp", "ensemble", "cross_ensemble"],
                        default="cross_ensemble")
    common.add_argument("--pca", type=int, default=None,
                        help="Override PCA components (default: model-dependent)")
    common.add_argument("--force", action="store_true", help="Retrain even if artifact exists")

    subparsers.add_parser("download", help="Download FPBase data")
    subparsers.add_parser("embed", parents=[common], help="Generate embeddings")
    subparsers.add_parser("train", parents=[common], help="Train model")
    subparsers.add_parser("evaluate", parents=[common], help="Evaluate saved model")
    subparsers.add_parser("run", parents=[common], help="Full pipeline (default)")

    args = parser.parse_args()

    if args.command is None or args.command == "run":
        # Set defaults if run with no args
        if args.command is None:
            args.embedding = "esm2"
            args.model = "cross_ensemble"
            args.pca = None
            args.force = False
        cmd_run(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "embed":
        cmd_embed(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)


if __name__ == "__main__":
    main()
