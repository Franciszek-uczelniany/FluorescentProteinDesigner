# Fluorescent Protein Emission Predictor

Predicts the emission wavelength (in nanometers) of fluorescent proteins from their amino acid sequence. Uses ESM-2 protein language model embeddings with Ridge regression and MLP models trained on 839 proteins from FPBase.

**Best performance**: Ensemble model achieves ~21 nm mean absolute error.

## Setup

1. Install Python 3.10+ (recommend [miniconda](https://docs.conda.io/en/latest/miniconda.html)):
   ```bash
   conda create -n proteins python=3.12
   conda activate proteins
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) For ESM-C 600M support:
   ```bash
   pip install esm
   ```

## Quick start

Run the full pipeline (download data, generate embeddings, train, evaluate):

```bash
python pipeline.py
```

This takes ~10 minutes the first time (downloading model + embedding 839 sequences). Subsequent runs take seconds because everything is cached.

## Predicting on your own sequences

After training, predict emission wavelength for any amino acid sequence:

```bash
python predict.py artifacts/esm2/ensemble.json --sequence "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKL"
```

For multiple sequences from a FASTA file:

```bash
python predict.py artifacts/esm2/ensemble.json --fasta candidates.fasta
```

As a Python library (for search algorithms):

```python
from predict import EmissionPredictor

predictor = EmissionPredictor("artifacts/esm2/ensemble.json")
wavelength = predictor.predict("MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKL")
# -> 509.3

wavelengths = predictor.predict_batch(["MVSK...", "MEEL...", ...])
# -> np.array([509.3, 527.1, ...])
```

## Trying different models

| Command | What it does |
|---------|-------------|
| `python pipeline.py` | Train ensemble (Ridge + MLP) with ESM-2 embeddings |
| `python pipeline.py run --model ridge` | Ridge regression only |
| `python pipeline.py run --model mlp` | MLP only |
| `python pipeline.py run --embedding esmc600m` | Use ESM-C 600M instead of ESM-2 |
| `python pipeline.py train --model ridge --pca 128` | Ridge with custom PCA dimensions |

Individual steps:
```bash
python pipeline.py download           # just download FPBase data
python pipeline.py embed              # just generate embeddings
python pipeline.py train              # just train models
python pipeline.py evaluate           # just evaluate saved models
```

## What the output means

- **MAE** (Mean Absolute Error): Average prediction error in nanometers. Lower is better. Our ensemble achieves ~21 nm, meaning predictions are off by ~21 nm on average.
- **RMSE** (Root Mean Squared Error): Like MAE but penalizes large errors more. Useful for spotting outliers.
- **R2** (R-squared): How much variance the model explains. 1.0 = perfect, 0.0 = no better than predicting the mean. Our models achieve ~0.85.

Fluorescent protein emission wavelengths range from ~380 nm (UV) to ~1000 nm (near-IR), so a 21 nm error is quite good for most applications.

## Project structure

```
pipeline.py       CLI entry point
predict.py        Inference script + importable library
data.py           FPBase download + caching + train/test split
embeddings.py     ESM-2 and ESM-C embedding generation
models.py         Model definitions, training, saving, loading
requirements.txt  Dependencies

cache/            Auto-created, cached intermediate data
artifacts/        Saved trained models
_old/             Previous scripts (for reference)
```

## Troubleshooting

**Out of memory during embedding**: ESM-2 (650M parameters) needs ~3 GB RAM. Close other applications or use a machine with more memory.

**MPS (Apple Silicon) errors**: If you see MPS-related errors, set `PYTORCH_ENABLE_MPS_FALLBACK=1` or force CPU:
```bash
CUDA_VISIBLE_DEVICES="" python pipeline.py
```

**Missing `esm` package**: ESM-C support requires `pip install esm`. ESM-2 works without it (uses HuggingFace transformers).

**Want to re-embed**: Delete the relevant `cache/esm2/` or `cache/esmc600m/` directory and re-run.

**Want to retrain**: Use the `--force` flag: `python pipeline.py train --force`
