# Deepfake Image Forensics Lab

An image-forensics experiment that classifies face images as **real** or **fake** with transfer learning. The notebook trains and evaluates three PyTorch classifiers:

1. a frozen ImageNet **ResNet-50** backbone with a binary head;
2. a frozen ImageNet **ViT-B/16** backbone with a binary head; and
3. a **CNN + ViT feature-fusion** classifier.

The project includes the labeled image set, exploratory checks, stratified train/validation/test splitting, model training, evaluation, error analysis, and inference on unseen images.

## Results

Results stored in the executed notebook use a fixed `random_state=42` stratified 70/15/15 split. The held-out test set contains 194 images (89 real and 105 fake).

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| ResNet-50 | 95.88% | 93.69% | 99.05% | 96.30% |
| ViT-B/16 | 98.97% | 98.13% | 100.00% | 99.06% |
| CNN + ViT fusion | **99.48%** | **99.06%** | **100.00%** | **99.53%** |

These figures describe this particular dataset and split. They should not be interpreted as performance on deepfakes from unseen generators, identities, or compression pipelines.

## Dataset

The included dataset contains 1,289 labeled JPEG images:

| Class | Images |
|---|---:|
| Real | 589 |
| Fake | 700 |

Labels and image IDs are listed in `dataset/data.csv`. Images are arranged under `dataset/real/` and `dataset/fake/`. Three additional images in `fakeImages/` are used by the final inference example.

Before republishing the dataset, verify that you have the necessary redistribution rights and add the original dataset source/citation here. The repository does not currently contain provenance or a dataset license.

## Repository layout

```text
.
|-- deepfake_detection.ipynb   # Complete analysis, training, and evaluation
|-- requirements.txt           # Python dependencies
|-- dataset/
|   |-- data.csv               # image ID and label
|   |-- real/                  # authentic images
|   `-- fake/                  # manipulated images
`-- fakeImages/                # images used for inference demonstration
```

## Run locally

Python 3.10+ is recommended. A CUDA-capable GPU is strongly recommended because the notebook trains both ResNet-50 and ViT-B/16 models.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <REPOSITORY_FOLDER>

python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
jupyter notebook deepfake_detection.ipynb
```

The pretrained torchvision weights are downloaded on the first run, so that run requires internet access or a populated PyTorch cache.

Run the notebook from the repository root so its relative `dataset/` and `fakeImages/` paths resolve correctly. Re-running all cells can take substantial time and GPU memory.

## Method summary

- Images are normalized with ImageNet channel statistics.
- Data is split with class stratification and a fixed seed.
- Each pretrained backbone is frozen; only its binary classification head is trained.
- The fusion network concatenates 2,048 ResNet features and 768 ViT features, then trains a small regularized binary classifier.
- Predictions use a sigmoid probability with a 0.5 threshold (`fake = 1`, `real = 0`).
- Evaluation includes loss, accuracy, precision, recall, F1, confusion matrices, classification reports, and misclassified-image inspection.

## Reproducibility notes

The data split is deterministic, but training is not fully deterministic because global random seeds and deterministic PyTorch algorithms are not configured. Results may vary by hardware and software version. No trained checkpoints are committed; running the notebook trains models in memory.

## Responsible use

This is an educational prototype, not a production forensic system. Do not use its output as the sole basis for moderation, attribution, legal, employment, or identity decisions. Real-world deployment requires evaluation on representative external data, calibration, bias analysis, provenance checks, and human review.W