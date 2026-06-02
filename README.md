# Fake News Detection

PyTorch project for multimodal fake news detection with image, text, and metadata inputs. The current training pipeline uses a three-tower model:

- Image tower: ResNet50 pretrained on ImageNet
- Text tower: BERT `bert-base-uncased`
- Metadata tower: MLP over `score`, `num_comments`, and `upvote_ratio`

The project also supports ablation experiments and exports analysis artifacts for later explainable AI work.

## Project Structure

```text
src/
  build_mapping.py              # Build image/text metadata mapping from raw Fakeddit-style files
  prepare_data.py               # Prepare train/val/test multimodal CSVs and image folders
  train_multimodal.py           # Train, validate, test, and export analysis artifacts
  run_ablation_experiments.py   # Run modality ablation experiments sequentially
  visualization_english.py      # Dataset visualization utilities

processed/
  mapping/                      # Lightweight mapping CSVs
  multimodal_training/          # Train/val/test CSV metadata

output/
  figures/                      # Generated visualization figures
```

Large raw data, processed images, model checkpoints, and analysis outputs are intentionally ignored by Git. See `.gitignore`.

## Environment

Python 3.11 is recommended. On macOS with conda:

```bash
conda create -n fakenewsdetection python=3.11 -y
conda activate fakenewsdetection
cd /Users/bessie/Desktop/fake_news_detection
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

If Hugging Face downloads are slow or unstable in China, use:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## Data

The training CSVs are expected at:

```text
processed/multimodal_training/train.csv
processed/multimodal_training/val.csv
processed/multimodal_training/test.csv
```

Images are expected at:

```text
processed/multimodal_training/images/train/
processed/multimodal_training/images/val/
processed/multimodal_training/images/test/
```

The image folders are not tracked because they are large. To rebuild processed data locally, place the raw dataset under `data/` and run:

```bash
python src/build_mapping.py
python src/prepare_data.py
```

## Training

Train the full multimodal model:

```bash
conda activate fakenewsdetection
export HF_ENDPOINT=https://hf-mirror.com

python src/train_multimodal.py --epochs 5 --batch-size 2 --log-every 20
```

The script automatically chooses the best available device:

```text
cuda -> mps -> cpu
```

No hard-coded `.cuda()` calls are used.

## Model

`TriTowerFakeNewsModel` combines:

- ResNet50 visual features projected to 512 dimensions
- BERT `[CLS]` text features projected to 512 dimensions
- Metadata features projected to 128 dimensions

The fused feature is:

```text
512 image + 512 text + 128 metadata = 1152
```

The classifier uses dropout and outputs binary logits.

## Class Imbalance and Metrics

The dataset is imbalanced, with roughly:

```text
fake: ~65%
real: ~35%
```

`train_multimodal.py` dynamically computes inverse-frequency class weights from the training set and applies them to `CrossEntropyLoss`.

Validation and test evaluation report:

- Accuracy
- Macro F1
- Precision / Recall / F1 classification report
- Confusion matrix

Best checkpoints are selected by validation Macro F1.

## Ablation Experiments

Each modality can be enabled or disabled:

```bash
--use-image / --no-use-image
--use-text  / --no-use-text
--use-meta  / --no-use-meta
```

Run all ablations sequentially with batch size 32:

```bash
python src/run_ablation_experiments.py --epochs 5 --batch-size 32 --log-every 20
```

Checkpoints are saved separately under `best_models/`, for example:

```text
best_models/best_image_text_meta.pth
best_models/best_image.pth
best_models/best_text.pth
best_models/best_meta.pth
```

## Analysis Outputs

After training, the script loads the best checkpoint and runs test-set analysis. Outputs are saved under:

```text
output/analysis/
```

Generated files include:

- `confusion_matrix.png`
- `cosine_similarity_dist.csv`
- `sample_1_probes.pt` through `sample_5_probes.pt`

Probe files contain:

- image ID
- text
- label and prediction
- ResNet layer4 feature map
- BERT last-layer attention
- image-text cosine similarity

These files are ignored by Git because they are generated artifacts.

## Notes

The current split is random and may overestimate performance because subreddit/source style can leak label information through images, titles, and metadata distributions. A stricter future evaluation should use group-based splitting by subreddit or source.
