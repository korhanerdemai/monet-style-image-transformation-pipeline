# Monet-Style Image Transformation Pipeline

**Author:** Erdem Korhan Erdem
**GitHub Repository:** [monet-style-image-transformation-pipeline](https://github.com/korhanerdemai/monet-style-image-transformation-pipeline)

---

## 1. Project Overview & Conceptual Content

### Problem Statement

The goal of this project is to build an end-to-end MLOps pipeline that performs unpaired image-to-image translation, specifically converting real-world landscape photographs into paintings that mimic the style of Claude Monet.

Unpaired style transfer is traditionally difficult because it requires a nuanced understanding of texture, brushwork, and lighting without matching content-style image pairs. This project provides a robust, reproducible codebase that uses a baseline feed-forward model alongside a main adversarial model to solve this problem while validating using a distance-based quality metric.

### Input and Output Data Format

- **Input:** RGB landscape photographs.
  - **Structure:** 4D Tensor `(Batch, Height, Width, Channels)`
  - **Dimensionality:** `(N, 256, 256, 3)`
- **Output:** RGB images in the style of Claude Monet.
  - **Structure:** 4D Tensor `(Batch, Height, Width, Channels)`
  - **Dimensionality:** `(N, 256, 256, 3)`

---

## 2. Evaluation System

### Metrics

We track five distinct metrics spanning qualitative, quantitative, and industrial dimensions:

1.  **Memorization-informed Fréchet Inception Distance (MiFID) [Primary Metric]:**
    - _Target:_ `< 50.0` (initial target), `< 40.0` (long-term goal).
    - _Purpose:_ MiFID measures both the stylistic quality of generated images (using an InceptionV3 backbone) and penalizes the generator if it simply "memorizes" and copies the training images. Lower is better.
2.  **Fréchet Inception Distance (FID):**
    - _Target:_ `< 60.0`
    - _Purpose:_ Standard baseline metric for generative adversarial networks to evaluate style transformation quality.
3.  **Cycle-Consistency Reconstruction Error (L1 Loss):**
    - _Target:_ Decreasing trend during training.
    - _Purpose:_ Measures content preservation by calculating the mean absolute difference between a photo and its reconstructed version after a round-trip mapping: `Photo -> Monet -> Photo`.
4.  **Identity Error (L1 Loss):**
    - _Target:_ Decreasing trend during training.
    - _Purpose:_ Measures color and style stability. It measures the reconstruction loss when feeding a real Monet painting into the Monet generator (which should remain unchanged).
5.  **Inference Latency (Milliseconds per Image):**
    - _Target:_ `< 500 ms/image` on standard deployment hardware.
    - _Purpose:_ Operational engineering metric to ensure the model can be served as a production REST API.

### Validation Strategy

Because unpaired datasets lack ground-truth target images, standard supervised validation is not possible. Instead, we use intermediate metric evaluations on a dedicated split:

- **Train Set:** `5,000` landscape photos and `250` Monet paintings used strictly for backpropagation.
- **Validation Set:** `1,028` landscape photos and `50` Monet paintings used to calculate MiFID, FID, and Cycle-Consistency Loss during training for hyperparameter tuning.
- **Test Set:** `1,000` landscape photos held out for final inference testing.

### Reproducibility

- **Data manifests:** Rather than generating splits dynamically, explicit train/val/test CSV manifest files are generated once and tracked using **DVC (Data Version Control)**.
- **Environment:** Encapsulated software dependencies via `uv` virtual environments and Docker images.

---

## 3. Modeling & Data Infrastructure

### Data Source

- **Source:** [Kaggle GAN Getting Started Competition](https://www.kaggle.com/competitions/gan-getting-started)
- **Composition:** `300` Monet paintings and `7,028` landscape photographs (256x256 resolution).
- **Mitigation of Overfitting:** The extremely limited size of the Monet domain (300 paintings) is mitigated via random horizontal flips, color jitter, and identity preservation loss.

### Model Architectures

- **Baseline (AdaIN Style Transfer):**
  - A feed-forward Adaptive Instance Normalization (AdaIN) baseline network following Huang & Belongie.
  - Uses a pretrained ConvNeXt backbone for hierarchical style and content representation extraction, coupled with a trained decoder.
- **Main Model (CycleGAN):**
  - A dual-generator (ResNet or U-Net) and dual-discriminator (PatchGAN) framework.
  - Trained with a weighted combination of Least Squares GAN (LSGAN) loss, Cycle-Consistency loss, and Identity loss.

---

## 4. Onboarding & Technical Details

### Project Structure

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

### Setup

Follow these steps to configure your local development environment:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/korhanerdemai/monet-style-image-transformation-pipeline.git
    cd monet-style-image-transformation-pipeline
    ```

2.  **Install `uv`:**
    If you don't have `uv` installed, run:

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

3.  **Sync Dependencies:**
    Use `uv` to automatically provision a virtual environment and synchronize dependencies:

    ```bash
    uv sync
    ```

4.  **Pull Data via DVC (Optional):**
    The dataset is automatically downloaded and extracted for you the first time you run any training or inference command. However, if you'd like to manually download the raw images and pre-split manifest CSV files upfront:

    ```bash
    uv run dvc pull
    ```

    Once the dataset is pulled and automatically extracted, the directory structure under `data/raw/` will be structured as follows:

    ```
    data/raw/monet_dataset/
    ├── monet_jpg/              # 300 Monet paintings
    ├── monet_tfrec/            # TFRecords of Monet paintings
    ├── photo_jpg/              # 7,028 landscape photographs
    ├── photo_tfrec/            # TFRecords of photographs
    ├── train_manifest.csv      # Pre-generated train split CSV
    ├── val_manifest.csv        # Pre-generated validation split CSV
    └── test_manifest.csv       # Pre-generated test split CSV
    ```

---

### Train & Preprocessing Pipeline

Training is organized into distinct phases via modular entry points exposed by `commands.py`.

#### Dataset Manifest Splits

The deterministic train-val-test split manifests (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`) are pre-generated and included directly in the DVC-pulled raw dataset folder. Therefore, developers **do not need to execute any dataset splitting scripts** before training.

_(If you ever need to recreate or modify the splits, you can inspect or run the utility script `scripts/make_dataset_splits.py`)._

#### Configuration Management

The training pipeline uses **Hydra** to manage hyperparameters, model architecture choices, data resolutions, and logging configurations. These settings are stored under the `configs/` directory:

1.  **Training Parameters (`configs/training/default.yaml`):**
    - `epochs`: Total number of training epochs (default: `18`).
    - `decay_epochs`: Number of epochs over which the learning rate decays linearly (default: `18`).
    - `batch_size`: Batch size used by the data loaders (default: `1`).
    - `learning_rate`: Generator and discriminator learning rate (default: `0.0002`).
    - `fast_dev_run`: Set to `true` to run a 1-batch sanity check (default: `false`).
2.  **CycleGAN Model Parameters (`configs/model/cyclegan.yaml`):**
    - `gen_name`: Generator architecture type — choose `"unet"` or `"resnet"` (default: `"unet"`).
    - `num_resblocks`: Number of residual blocks used if utilizing the `resnet` generator (default: `6`).
    - `hid_channels`: Hidden dimension channel size of the first conv layer (default: `64`).
    - `lambda_idt`: Identity loss multiplier weighting coefficient (default: `0.5`).
    - `lambda_cycle`: Cycle-consistency loss weights for Monet and Photo domains (default: `[10.0, 10.0]`).
3.  **AdaIN Baseline Parameters (`configs/model/baseline.yaml`):**
    - `decoder_lr`: Adam optimizer learning rate for baseline decoder (default: `0.0001`).
    - `style_weight`: Weighting multiplier for the style/texture loss component (default: `10.0`).
4.  **Logging Parameters (`configs/logging/default.yaml`):**
    - `tracking_uri`: Tracking URL where MLflow metrics are registered (default: `"http://127.0.0.1:8080"`).
    - `experiment_name`: Project experiment identifier (default: `"CycleGAN_Monet"`).

To adjust parameters, you can modify the respective `.yaml` configuration files inside the `configs/` directory.

---

#### 1. Training the AdaIN Baseline

To train the baseline decoder on top of the frozen ConvNeXt encoder:

```bash
uv run python commands.py train_baseline
```

- **CLI Overrides:** You can override parameters directly in the command line instead of editing the baseline configuration file:
  ```bash
  uv run python commands.py train_baseline --epochs 20 --batch_size 8 --max_steps 100
  ```

#### 2. Training the CycleGAN Model

To start the adversarial training of the main CycleGAN model (which runs via PyTorch Lightning):

```bash
uv run python commands.py train
```

- **Sanity Check (Fast Dev Run):** To quickly test the training loop with 1 training batch:
  ```bash
  uv run python commands.py train --fast_dev_run True
  ```
- **Experiment Tracking:** Training metrics and losses are automatically logged to the MLflow tracking server specified in `configs/logging/default.yaml`.

---

### Inference & Evaluation

#### 1. Fast Baseline Inference (Single Image Stylization)

To stylize an individual photograph using the trained AdaIN baseline:

```bash
uv run python commands.py run_baseline \
    --content_path data/raw/monet_dataset/photo_jpg/000ded5c7c.jpg \
    --style_path data/raw/monet_dataset/monet_jpg/000c1e65af.jpg \
    --output_path stylized_output.jpg
```

#### 2. Run Baseline Evaluation

To run the automated baseline evaluation script, which stylizes the validation/test set and saves performance metrics:

```bash
# Run a quick smoke test on 5 photos
uv run python -m monet_pipeline.evaluation.evaluate_baseline --n_photos 5 --verbose False

# Run evaluation on the complete dataset
uv run python -m monet_pipeline.evaluation.evaluate_baseline
```

Results will be written to `metrics/baseline_metrics.json`.

#### 3. Compute MiFID Separately

To compute the primary MiFID metric between any two custom image directories:

```bash
uv run python -m monet_pipeline.evaluation.metrics \
    --generated_dir path/to/stylized_images \
    --reference_dir data/raw/monet_dataset/monet_jpg \
    --batch_size 32
```
