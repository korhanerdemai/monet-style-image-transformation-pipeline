# Monet-Style Image Transformation Pipeline

> **CycleGAN-based MLOps pipeline** for translating landscape photographs into Monet-style paintings.

## Project Structure

```
monet-style-image-transformation-pipeline/
├── data/
│   ├── raw/               # DVC-tracked raw dataset (monet_dataset/)
│   └── processed/         # Processed / cached outputs
├── models/                # Saved model checkpoints
├── notebooks/             # Exploratory notebooks
│   └── cyclegan-implementation.ipynb
├── src/
│   ├── data/
│   │   └── data_loader.py # Modular PyTorch data pipeline
│   └── models/            # Model architecture modules (Phase 2)
├── tests/                 # Unit tests
├── pyproject.toml         # Project dependencies (uv)
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
from src.data.data_loader import CycleGANDataModule

dm = CycleGANDataModule.from_data_root("data/raw/monet_dataset", batch_size=1)
dm.setup("fit")
loader = dm.train_dataloader()
```

## Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **1** | ✅ Complete | Environment & Data Infrastructure |
| **2** | 🔜 Planned  | Model Architecture & Training |
| **3** | 🔜 Planned  | Experiment Tracking (MLflow/W&B) |
| **4** | 🔜 Planned  | Containerized Deployment |

## Data Versioning (DVC)

Raw data is tracked by DVC. The `.dvc` files are committed to git; actual image data
lives in the configured DVC remote.

```bash
dvc status          # Check if data is in sync
dvc pull            # Pull data from remote
dvc push            # Push new data to remote
```
