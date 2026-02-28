"""Dataset download, caching, and train/test split for FPBase fluorescent proteins."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL = "https://www.fpbase.org/api"
PROTEINS_URL = f"{BASE_URL}/proteins/?format=json"
SPECTRA_URL = f"{BASE_URL}/proteins/spectra/?format=json"

CACHE_DIR = Path(__file__).parent / "cache"


def _fetch_or_cache(url: str, cache_path: Path) -> list:
    """Fetch JSON from URL, caching to disk."""
    if cache_path.exists():
        print(f"  Using cached {cache_path.name}")
        with open(cache_path) as f:
            return json.load(f)
    print(f"  Downloading {url} ...")
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(data, f)
    print(f"  Saved {cache_path.name} ({len(data)} records)")
    return data


def download() -> None:
    """Download FPBase proteins and spectra JSON to cache/."""
    print("Downloading FPBase data...")
    _fetch_or_cache(PROTEINS_URL, CACHE_DIR / "fpbase_proteins.json")
    _fetch_or_cache(SPECTRA_URL, CACHE_DIR / "fpbase_spectra.json")
    print("  Done.\n")


def get_dataset() -> pd.DataFrame:
    """Return (name, sequence, em_max) DataFrame. Downloads + caches if needed.

    Cached as cache/fpbase_sequences.parquet after first run.
    """
    parquet_path = CACHE_DIR / "fpbase_sequences.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"Loaded {len(df)} proteins from cache ({parquet_path.name})")
        return df

    # Ensure raw JSON is downloaded
    proteins_path = CACHE_DIR / "fpbase_proteins.json"
    if not proteins_path.exists():
        download()

    print("Building dataset from FPBase JSON...")
    with open(proteins_path) as f:
        proteins = json.load(f)

    # Build lookup: slug -> (name, sequence, em_max)
    seq_lookup = {}
    for p in proteins:
        slug = p.get("slug") or p.get("name", "").lower().replace(" ", "-")
        seq = p.get("seq", "")
        if not seq or not slug:
            continue
        em_max = None
        for state in p.get("states", []):
            if state.get("em_max"):
                em_max = state["em_max"]
                break
        seq_lookup[slug] = {"name": p.get("name", slug), "sequence": seq, "em_max": em_max}

    # Filter to proteins with em_max
    rows = []
    for slug, info in seq_lookup.items():
        if info["em_max"] is None:
            continue
        rows.append({
            "name": info["name"],
            "sequence": info["sequence"],
            "em_max": info["em_max"],
        })

    df = pd.DataFrame(rows)
    print(f"  {len(df)} proteins with sequence + em_max")

    if df.empty:
        print("ERROR: No matching proteins found.")
        sys.exit(1)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved {parquet_path.name}")
    return df


def get_split(X: np.ndarray, y: np.ndarray, names: np.ndarray | None = None):
    """Canonical 80/20 split. Single source of truth for random_state=42.

    Returns (X_train, X_test, y_train, y_test) or
            (X_train, X_test, y_train, y_test, names_train, names_test) if names provided.
    """
    if names is not None:
        return train_test_split(X, y, names, test_size=0.2, random_state=42)
    return train_test_split(X, y, test_size=0.2, random_state=42)
