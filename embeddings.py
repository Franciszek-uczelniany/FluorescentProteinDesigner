"""ESM-2, ESM-C 600M, and ESM-C 6B embedding generation with augmented pooling.

Caches embeddings as individual .npy files under cache/{model_name}/.
ESM-C 6B uses the Forge API and supports incremental/resumable embedding.
"""

import json
import numpy as np
import os
import sys
import torch
from pathlib import Path

from data import CACHE_DIR, get_dataset

# ── ESM-2 config ──────────────────────────────────────────────────────────────
ESM2_MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
ESM2_HIDDEN_DIM = 1280
ESMC_HIDDEN_DIM = 1152
ESMC6B_HIDDEN_DIM = 2560
BATCH_SIZE = 8
MAX_LENGTH = 1022  # ESM-2 max tokens (excluding BOS/EOS)

SUPPORTED_MODELS = {"esm2", "esmc600m", "esmc6b"}


def get_embeddings(model_name: str = "esm2") -> dict:
    """Return {"mean": ndarray, "augmented": ndarray, "names": ndarray, "em_max": ndarray}.

    Loads from cache/{model_name}/ if exists, otherwise computes + caches.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {SUPPORTED_MODELS}")

    cache = CACHE_DIR / model_name
    mean_path = cache / "mean.npy"
    aug_path = cache / "augmented.npy"
    names_path = cache / "names.npy"
    em_max_path = cache / "em_max.npy"

    if all(p.exists() for p in [mean_path, aug_path, names_path, em_max_path]):
        print(f"Loading cached {model_name} embeddings...")
        return {
            "mean": np.load(mean_path),
            "augmented": np.load(aug_path),
            "names": np.load(names_path, allow_pickle=True),
            "em_max": np.load(em_max_path),
        }

    # ESM-C 6B uses incremental embedding with its own caching
    if model_name == "esmc6b":
        df = get_dataset()
        sequences = df["sequence"].tolist()
        names = df["name"].values
        em_max = df["em_max"].values

        mean_emb, aug_emb = _embed_esmc6b(sequences, names)

        cache.mkdir(parents=True, exist_ok=True)
        np.save(mean_path, mean_emb)
        np.save(aug_path, aug_emb)
        np.save(names_path, names)
        np.save(em_max_path, em_max)
        print(f"  Cached to {cache}/")

        return {"mean": mean_emb, "augmented": aug_emb, "names": names, "em_max": em_max}

    # Need to compute embeddings
    df = get_dataset()
    sequences = df["sequence"].tolist()
    names = df["name"].values
    em_max = df["em_max"].values

    if model_name == "esm2":
        mean_emb, aug_emb = _embed_esm2(sequences)
    else:
        mean_emb, aug_emb = _embed_esmc(sequences)

    # Cache
    cache.mkdir(parents=True, exist_ok=True)
    np.save(mean_path, mean_emb)
    np.save(aug_path, aug_emb)
    np.save(names_path, names)
    np.save(em_max_path, em_max)
    print(f"  Cached to {cache}/")

    return {"mean": mean_emb, "augmented": aug_emb, "names": names, "em_max": em_max}


def embed_single(sequence: str, model_name: str = "esm2") -> tuple[np.ndarray, np.ndarray]:
    """Embed a single sequence. Returns (mean, augmented) arrays of shape (1, D) and (1, 4D)."""
    if model_name == "esm2":
        return _embed_esm2([sequence])
    elif model_name == "esmc6b":
        return _embed_esmc6b_single(sequence)
    else:
        return _embed_esmc([sequence])


def _pool_hidden_states(hidden: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mean/std/min/max pooling over residue positions.

    Args:
        hidden: (batch, seq_len, hidden_dim)
        mask: (batch, seq_len) attention mask with BOS/EOS already zeroed out

    Returns:
        mean_emb: (batch, hidden_dim)
        augmented_emb: (batch, 4 * hidden_dim)
    """
    mask_expanded = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
    counts = mask_expanded.sum(dim=1).clamp(min=1)  # (batch, 1)

    # Mean pool
    summed = (hidden * mask_expanded).sum(dim=1)
    mean_emb = summed / counts

    # Std pool
    diff_sq = ((hidden - mean_emb.unsqueeze(1)) * mask_expanded) ** 2
    variance = diff_sq.sum(dim=1) / counts.clamp(min=2)
    std_emb = variance.sqrt()

    # Min/Max pool (set masked positions to +inf/-inf so they're ignored)
    big_val = 1e9
    hidden_for_min = hidden + (1 - mask_expanded) * big_val
    hidden_for_max = hidden + (1 - mask_expanded) * (-big_val)
    min_emb = hidden_for_min.min(dim=1).values
    max_emb = hidden_for_max.max(dim=1).values

    # Concatenate: mean | std | min | max
    augmented = torch.cat([mean_emb, std_emb, min_emb, max_emb], dim=1)

    return mean_emb, augmented


def _mask_bos_eos(mask: torch.Tensor) -> torch.Tensor:
    """Zero out BOS (position 0) and EOS (last non-padding token) in attention mask."""
    mask = mask.clone()
    mask[:, 0] = 0  # BOS
    for j in range(mask.shape[0]):
        if mask[j].any():
            eos_pos = mask[j].nonzero()[-1].item()
            mask[j, eos_pos] = 0
    return mask


def _embed_esm2(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Batch processing via HuggingFace transformers. Returns (mean, augmented)."""
    from transformers import AutoModel, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading ESM-2 ({ESM2_MODEL_NAME}) on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(ESM2_MODEL_NAME)
    model = AutoModel.from_pretrained(ESM2_MODEL_NAME).to(device)
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters)")

    all_mean = []
    all_augmented = []
    n_batches = (len(sequences) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(sequences), BATCH_SIZE):
        batch_seqs = sequences[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        print(f"  Batch {batch_num}/{n_batches} ({len(batch_seqs)} sequences)", end="")

        tokens = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH + 2,  # +2 for BOS/EOS
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokens)

        hidden = outputs.last_hidden_state
        mask = _mask_bos_eos(tokens["attention_mask"])

        mean_emb, augmented = _pool_hidden_states(hidden, mask)

        all_mean.append(mean_emb.cpu().numpy())
        all_augmented.append(augmented.cpu().numpy())
        print(f" -> mean {mean_emb.shape[1]}d, augmented {augmented.shape[1]}d")

    mean_embeddings = np.concatenate(all_mean, axis=0)
    augmented_embeddings = np.concatenate(all_augmented, axis=0)
    print(f"\n  Mean embeddings: {mean_embeddings.shape}")
    print(f"  Augmented embeddings: {augmented_embeddings.shape}")
    return mean_embeddings, augmented_embeddings


def _embed_esmc(sequences: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """One-at-a-time via esm package. Returns (mean, augmented)."""
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
    except ImportError:
        raise ImportError(
            "ESM-C requires the 'esm' package. Install with: pip install esm"
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # ESM-C may not support MPS well, fall back to CPU if needed
    if device == "mps":
        try:
            client = ESMC.from_pretrained("esmc_600m").to(device)
        except Exception:
            print("  MPS not supported for ESM-C, falling back to CPU")
            device = "cpu"
            client = ESMC.from_pretrained("esmc_600m").to(device)
    else:
        client = ESMC.from_pretrained("esmc_600m").to(device)

    print(f"Loaded ESM-C 600M on {device}")

    all_mean = []
    all_augmented = []

    for i, seq in enumerate(sequences):
        print(f"  Sequence {i + 1}/{len(sequences)}", end="\r")

        protein = ESMProtein(sequence=seq)
        tensor = client.encode(protein)
        output = client.logits(tensor, LogitsConfig(return_embeddings=True))
        hidden = output.embeddings  # (1, seq_len, 1152)

        # Create a mask of all ones (no padding for single sequences), exclude BOS/EOS
        seq_len = hidden.shape[1]
        mask = torch.ones(1, seq_len, device=hidden.device)
        mask = _mask_bos_eos(mask)

        mean_emb, augmented = _pool_hidden_states(hidden, mask)

        all_mean.append(mean_emb.cpu().numpy())
        all_augmented.append(augmented.cpu().numpy())

    print()
    mean_embeddings = np.concatenate(all_mean, axis=0)
    augmented_embeddings = np.concatenate(all_augmented, axis=0)
    print(f"  Mean embeddings: {mean_embeddings.shape}")
    print(f"  Augmented embeddings: {augmented_embeddings.shape}")
    return mean_embeddings, augmented_embeddings


# ── ESM-C 6B (Forge API) ─────────────────────────────────────────────────────

def _load_api_token() -> str:
    """Load EVOLUTIONARY_SCALE_API_KEY from env var or .env file."""
    token = os.environ.get("EVOLUTIONARY_SCALE_API_KEY")
    if token:
        return token

    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key == "EVOLUTIONARY_SCALE_API_KEY" and value:
                    return value

    print(
        "Error: EVOLUTIONARY_SCALE_API_KEY not found.\n"
        "Set it in .env or as an environment variable.\n"
        "Get your token from https://forge.evolutionaryscale.ai"
    )
    sys.exit(1)


def _get_forge_client():
    """Create an ESM-C 6B Forge API client."""
    try:
        from esm.sdk.forge import ESM3ForgeInferenceClient
    except ImportError:
        raise ImportError(
            "ESM-C 6B requires the 'esm' package. Install with: pip install esm"
        )

    token = _load_api_token()
    return ESM3ForgeInferenceClient(
        model="esmc-6b-2024-12",
        url="https://forge.evolutionaryscale.ai",
        token=token,
    )


def _embed_esmc6b(sequences: list[str], names: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Incremental embedding via Forge API. Saves progress after each sequence."""
    from esm.sdk.api import ESMProtein, LogitsConfig

    partial_dir = CACHE_DIR / "esmc6b" / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    progress_path = partial_dir / "progress.json"

    total = len(sequences)

    # Load existing progress
    completed = set()
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        completed = set(progress.get("completed", []))
        print(f"Resuming ESM-C 6B embedding from {len(completed)}/{total}...")
    else:
        print(f"Starting ESM-C 6B embedding ({total} sequences)...")

    remaining = [i for i in range(total) if i not in completed]

    if not remaining:
        print("  All sequences already embedded, consolidating...")
        return _consolidate_esmc6b(total)

    client = _get_forge_client()
    tokens_used = 0
    already_done = len(completed)

    for count, idx in enumerate(remaining):
        seq = sequences[idx]
        print(f"  Sequence {already_done + count + 1}/{total} "
              f"(idx={idx}, len={len(seq)})", end="")

        try:
            protein = ESMProtein(sequence=seq)
            tensor = client.encode(protein)
            output = client.logits(tensor, LogitsConfig(return_embeddings=True))
            hidden = output.embeddings  # (1, seq_len, 2560)

            seq_len = hidden.shape[1]
            mask = torch.ones(1, seq_len, device=hidden.device)
            mask = _mask_bos_eos(mask)

            mean_emb, augmented = _pool_hidden_states(hidden, mask)

            # Save per-sequence partial result
            result = np.concatenate(
                [mean_emb.cpu().numpy(), augmented.cpu().numpy()], axis=1
            )
            np.save(partial_dir / f"{idx}.npy", result)

            completed.add(idx)
            tokens_used += len(seq)

            # Update progress file
            progress_path.write_text(json.dumps({
                "completed": sorted(completed),
                "total": total,
            }))

            credits_est = tokens_used / 10000
            print(f" -> ok (~{credits_est:.1f} credits this session)")

        except Exception as e:
            error_msg = str(e)
            credits_est = tokens_used / 10000

            print(f" -> error: {error_msg}")
            print(f"\nEmbedded {len(completed)}/{total} sequences "
                  f"(~{credits_est:.1f} credits used this session).")

            if "429" in error_msg or "rate" in error_msg.lower():
                print("Rate limit hit. Wait a moment and try again.")
            elif "credit" in error_msg.lower() or "quota" in error_msg.lower():
                print("Credit limit reached. Run again tomorrow to continue.")
            else:
                print(f"Run again to resume from sequence {len(completed) + 1}.")

            sys.exit(1)

    print(f"\n  All {total} sequences embedded! Consolidating...")
    return _consolidate_esmc6b(total)


def _consolidate_esmc6b(total: int) -> tuple[np.ndarray, np.ndarray]:
    """Consolidate partial .npy files into final mean + augmented arrays."""
    partial_dir = CACHE_DIR / "esmc6b" / "partial"

    all_mean = []
    all_augmented = []
    mean_dim = ESMC6B_HIDDEN_DIM
    aug_dim = ESMC6B_HIDDEN_DIM * 4

    for idx in range(total):
        data = np.load(partial_dir / f"{idx}.npy")
        all_mean.append(data[:, :mean_dim])
        all_augmented.append(data[:, mean_dim:mean_dim + aug_dim])

    mean_embeddings = np.concatenate(all_mean, axis=0)
    augmented_embeddings = np.concatenate(all_augmented, axis=0)

    # Clean up partial files
    import shutil
    shutil.rmtree(partial_dir)

    print(f"  Mean embeddings: {mean_embeddings.shape}")
    print(f"  Augmented embeddings: {augmented_embeddings.shape}")
    return mean_embeddings, augmented_embeddings


def _embed_esmc6b_single(sequence: str) -> tuple[np.ndarray, np.ndarray]:
    """Embed a single sequence via ESM-C 6B Forge API."""
    from esm.sdk.api import ESMProtein, LogitsConfig

    client = _get_forge_client()

    protein = ESMProtein(sequence=sequence)
    tensor = client.encode(protein)
    output = client.logits(tensor, LogitsConfig(return_embeddings=True))
    hidden = output.embeddings  # (1, seq_len, 2560)

    seq_len = hidden.shape[1]
    mask = torch.ones(1, seq_len, device=hidden.device)
    mask = _mask_bos_eos(mask)

    mean_emb, augmented = _pool_hidden_states(hidden, mask)

    return mean_emb.cpu().numpy(), augmented.cpu().numpy()
