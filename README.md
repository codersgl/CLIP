# CLIP Reproduce (PyTorch)

This repository is a PyTorch reproduction of CLIP training. Current defaults:

- Visual encoder: `timm` `resnet50`
- Text encoder: HuggingFace `distilbert-base-uncased`
- Dataset: Flickr8k image-text pairs
- Config system: Hydra

## 1. Project Structure

```text
configs/                 # Hydra configs (model/data/optimizer)
scripts/train.py         # Training entrypoint
src/clip/                # Core code (model/loss/engine/data)
tests/                   # Unit tests
data/                    # Local dataset folder (Images + captions.txt)
outputs/                 # Hydra run outputs
```

## 2. Environment Setup

Requires Python >= 3.13 (see `pyproject.toml`).

### Option A: `uv` (recommended)

```bash
uv sync
```

### Option B: `pip`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 3. Data Preparation

Default paths (see `configs/data/Flickr8k.yaml`):

- Image directory: `data/Images`
- Annotation file: `data/captions.txt`

`captions.txt` must be a **comma-separated** text file. The code skips the first line and reads records like:

```text
image_name,caption
1000268201_693b08cb0e.jpg,A child in a pink dress is climbing up a set of stairs.
...
```

## 4. Start Training

Run from the repository root:

```bash
uv run python scripts/train.py
```

Or (in a `pip` environment):

```bash
python scripts/train.py
```

## 5. Common Hydra Overrides

```bash
# Change batch size and number of epochs
uv run python scripts/train.py data.batch_size=32 training.epochs=10

# Switch visual backbone
uv run python scripts/train.py model.visual_encoder.backbone_name=resnet18

# Set checkpoint path and log directory
uv run python scripts/train.py training.save_path=checkpoints/best.pt training.log_dir=tensorboard
```

## 6. Outputs and Logging

Because `hydra.job.chdir=True`, each run writes results under `outputs/YYYY-MM-DD/HH-MM-SS/`:

- `best.pt` (or your overridden `training.save_path`)
- `tensorboard/` (or your overridden `training.log_dir`)
- Hydra-generated config and runtime metadata

To view TensorBoard:

```bash
tensorboard --logdir outputs
```

## 7. Run Tests

```bash
uv run pytest -q
```

## 8. Inference (Retrieval)

Inference supports both directions:

- `text2image`: retrieve images for each text query
- `image2text`: retrieve texts for each image query
- default visibility: tqdm progress + runtime summary + matplotlib figure output
- fault tolerance: invalid images / empty texts are skipped and reported in stderr

You must provide the same training config family (`configs/`) to keep tokenizer and encoder settings consistent with checkpoint.

```bash
uv run python scripts/infer.py \
	--checkpoint outputs/2026-03-01/15-21-19/best.pt \
	--mode text2image \
	--images data/Images/1000268201_693b08cb0e.jpg data/Images/1001773457_577c3a7d70.jpg \
	--texts "a child climbing stairs" "a dog is running" \
	--topk 2 \
	--figure-dir inference_figures \
	--config-path configs \
	--config-name config
```

Stdout output remains JSON (top-k matches and scores) for compatibility.
Stderr prints summary/errors, and figure files are written under `--figure-dir`.

## 9. Current Default Config

- `embed_dim=512`
- `init_temperature=0.07`
- `optimizer=AdamW(lr=1e-4, betas=(0.9, 0.98), weight_decay=1e-3)`
- `epochs=20`
- `early stopping: patience=5`

You can adjust these values in `configs/` as needed.
