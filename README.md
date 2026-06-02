# Monet-Style Image Transformation Pipeline

> **CycleGAN-based MLOps pipeline** for translating landscape photographs into Monet-style paintings.

## Project Structure

```
monet-style-image-transformation-pipeline/
├── configs/               # Hydra configurations
├── data/
│   ├── raw/               # DVC-tracked raw dataset (monet_dataset/)
│   └── processed/         # Processed / cached outputs
├── metrics/               # Persisted evaluation results (JSON)
├── models/                # Saved model checkpoints
├── notebooks/             # Exploratory notebooks
│   └── cyclegan-implementation.ipynb
├── monet_pipeline/
│   ├── data/
│   │   └── data_loader.py          # Modular PyTorch data pipeline
│   ├── evaluation/
│   │   ├── metrics.py              # MiFID metric (PyTorch / InceptionV3)
│   │   └── evaluate_baseline.py   # End-to-end baseline evaluation script
│   └── models/
│       ├── baseline_adain.py      # ConvNeXt/AdaIN Style Transfer baseline
│       ├── cyclegan.py            # CycleGAN model structure
│       └── losses.py              # Custom loss definitions
├── scripts/               # Helper scripts
│   └── make_dataset_splits.py     # Deterministic dataset splitting script
├── tests/                 # Unit tests
├── pyproject.toml         # Project dependencies (uv)
├── commands.py            # Main CLI entry point
└── README.md
```

## Quick Start

### 1. Install dependencies (uv)

```bash
uv sync
```

### 2. Pull the DVC-tracked dataset

```bash
dvc pull
```

### 3. Use the data loader

```python
from monet_pipeline.data.data_loader import CycleGANDataModule

dm = CycleGANDataModule(
    train_manifest="data/processed/train_manifest.csv",
    val_manifest="data/processed/val_manifest.csv",
    batch_size=1,
)
dm.setup("fit")
loader = dm.train_dataloader()
```

### 4. Run the AdaIN baseline evaluation

```bash
# Quick smoke test (5 photos)
uv run python -m monet_pipeline.evaluation.evaluate_baseline --n_photos 5 --verbose False

# Full baseline run
uv run python -m monet_pipeline.evaluation.evaluate_baseline
```

Results are printed to the console and saved to `metrics/baseline_metrics.json`.

### 5. Stylize a single image (AdaIN)

```bash
uv run python commands.py run_baseline \
    --content_path data/raw/monet_dataset/photo_jpg/<photo>.jpg \
    --style_path   data/raw/monet_dataset/monet_jpg/<painting>.jpg \
    --output_path  stylized.jpg
```

### 6. Compute MiFID between two directories

```bash
uv run python -m monet_pipeline.evaluation.metrics <generated_dir> <reference_dir> --batch_size 32
```

## Phases

| Phase | Status      | Description                       |
| ----- | ----------- | --------------------------------- |
| **1** | ✅ Complete | Environment & Data Infrastructure |
| **2** | ✅ Complete | Baseline & Metrics (NST + MiFID)  |
| **3** | 🔜 Planned  | Experiment Tracking (MLflow/W&B)  |
| **4** | 🔜 Planned  | Containerized Deployment          |

## Evaluation

### Baseline — Neural Style Transfer (VGG-19)

The Phase 2 baseline uses Gatys et al. (2015) Neural Style Transfer via a frozen
VGG-19 feature extractor and L-BFGS optimisation. Default: **150 iterations per image**.

### Metric — MiFID

**Memorization-informed Fréchet Inception Distance** (MiFID) is the primary quality metric:

```
MiFID = FID / (cosine_memorization_distance + ε)
```

- **FID** measures distributional similarity between generated and real Monet features
  (InceptionV3 pool_3, 2048-d).
- **Cosine memorization distance** penalises generators that copy the training set;
  distances ≥ 0.1 are clamped to 1.0 (no penalty).
- Lower MiFID = better. A memorizing generator gets a very small denominator → score explodes.

## Data Versioning (DVC)

Raw data is tracked by DVC. The `.dvc` files are committed to git; actual image data
lives in the configured DVC remote (`D:\dvc-storage` by default).

```bash
dvc status          # Check if data is in sync
dvc pull            # Pull data from remote
dvc push            # Push new data to remote
```
