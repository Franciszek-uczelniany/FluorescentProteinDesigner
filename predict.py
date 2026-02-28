#!/usr/bin/env python
"""Inference script for fluorescent protein emission wavelength prediction.

CLI usage:
    python predict.py artifacts/esm2/ensemble.json --sequence "MVSKGEE..."
    python predict.py artifacts/esm2/ensemble.json --fasta candidates.fasta

Library usage:
    from predict import EmissionPredictor
    predictor = EmissionPredictor("artifacts/esm2/ensemble.json")
    wavelength = predictor.predict("MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKL")
    wavelengths = predictor.predict_batch(["MVSK...", "MEEL...", ...])
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import embeddings
import models


class EmissionPredictor:
    """Self-contained predictor that loads an artifact and handles embedding + inference."""

    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path)
        self.artifact = models.load_artifact(self.artifact_path)
        self._is_cross = (
            self.artifact["model_type"] == "ensemble"
            and "cross" in str(self.artifact.get("embedding", ""))
        )

    def predict(self, sequence: str) -> float:
        """Predict emission wavelength (nm) for a single amino acid sequence."""
        return float(self.predict_batch([sequence])[0])

    def _embed_sequences(self, sequences: list[str], model_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Embed sequences with the given model. Returns (mean, augmented)."""
        all_mean, all_aug = [], []
        for seq in sequences:
            m, a = embeddings.embed_single(seq, model_name)
            all_mean.append(m)
            all_aug.append(a)
        return np.concatenate(all_mean, axis=0), np.concatenate(all_aug, axis=0)

    def predict_batch(self, sequences: list[str]) -> np.ndarray:
        """Predict emission wavelengths (nm) for multiple sequences."""
        if self._is_cross:
            return self._predict_cross(sequences)

        # Single-embedding model
        emb_name = self.artifact["embedding"]
        mean_emb, aug_emb = self._embed_sequences(sequences, emb_name)

        if self.artifact["model_type"] == "ensemble":
            X = {"mean": mean_emb, "augmented": aug_emb}
        else:
            pooling = self.artifact.get("pooling", "mean")
            X = mean_emb if pooling == "mean" else aug_emb

        return models.predict(self.artifact, X)

    def _predict_cross(self, sequences: list[str]) -> np.ndarray:
        """Handle cross-embedding ensemble: each component uses its own embedding model."""
        components = self.artifact["components"]
        weights = self.artifact["weights"]
        preds = []

        for comp, w in zip(components, weights):
            emb_name = comp["embedding"]
            pooling = comp["pooling"]
            mean_emb, aug_emb = self._embed_sequences(sequences, emb_name)
            X = mean_emb if pooling == "mean" else aug_emb
            preds.append(models.predict(comp, X) * w)

        return sum(preds)


def _read_fasta(path: str) -> list[tuple[str, str]]:
    """Read sequences from a FASTA file. Returns list of (name, sequence)."""
    entries = []
    name = ""
    seq_parts = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if name and seq_parts:
                    entries.append((name, "".join(seq_parts)))
                name = line[1:].split()[0]
                seq_parts = []
            elif line:
                seq_parts.append(line)

    if name and seq_parts:
        entries.append((name, "".join(seq_parts)))

    return entries


def main():
    parser = argparse.ArgumentParser(
        description="Predict fluorescent protein emission wavelength"
    )
    parser.add_argument("artifact", help="Path to saved model artifact")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sequence", "-s", help="Amino acid sequence to predict")
    group.add_argument("--fasta", "-f", help="Path to FASTA file with sequences")

    args = parser.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        print(f"Artifact not found: {artifact_path}")
        sys.exit(1)

    predictor = EmissionPredictor(artifact_path)

    if args.sequence:
        wavelength = predictor.predict(args.sequence)
        print(f"Predicted emission: {wavelength:.1f} nm")
    else:
        entries = _read_fasta(args.fasta)
        if not entries:
            print(f"No sequences found in {args.fasta}")
            sys.exit(1)

        print(f"Predicting {len(entries)} sequences...\n")
        sequences = [seq for _, seq in entries]
        wavelengths = predictor.predict_batch(sequences)

        print(f"{'Name':<30s} {'Predicted (nm)':>14s}")
        print("-" * 46)
        for (name, _), wl in zip(entries, wavelengths):
            print(f"{name:<30s} {wl:>14.1f}")


if __name__ == "__main__":
    main()
