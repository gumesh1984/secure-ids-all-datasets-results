# Secure IDS: Multi-Dataset Reproducibility Package

This repository provides the implementation and experimental outputs for a leakage-safe, privacy-preserving intrusion-detection pipeline evaluated on four network-security datasets. The implementation combines a CNN-BiLSTM feature extractor, an MLP classifier, feature permutation, AES-GCM protection, and permissioned-blockchain integrity logging. It also contains the reviewer-requested reconstruction, membership-inference, attribute-inference, and ablation experiments.

## Repository contents

- `secure_ids_pipeline.py`: complete preprocessing, training, evaluation, privacy-attack, ablation, and result-packaging pipeline.
- `results/`: directly viewable principal CSV tables from the reported experiments.
- `reviewer_results_all_datasets.zip`: complete results archive, including all confusion-matrix figures and per-seed reviewer-experiment tables.
- `requirements.txt`: required Python packages.

Raw datasets and access credentials are intentionally excluded.

## Experimental configuration

| Setting | Value |
|---|---:|
| Random seeds | 42, 43, 44, 45, 46 |
| Test fraction | 0.30 |
| Training epochs | 20 |
| Batch size | 256 |
| Learning rate | 0.001 |
| Privacy-attack sample limit | 20,000 |
| Privacy-attack epochs | 10 |
| Shadow models | 2 |

The pipeline performs group-aware train/test splitting so duplicate feature vectors cannot cross the split. Imputation and scaling parameters are learned from the training partition only. The code audits target leakage before model fitting and records dataset preprocessing in `results/dataset_audit.csv`.

## Dataset preparation

Obtain the datasets from their original providers under their respective terms and place the CSV files in one directory. The reported run used the following dataset identities:

- CICIDS2017 binary balanced: 851,388 original rows and 53 columns.
- IoT Network Intrusion Dataset: 625,783 original rows and 86 columns.
- NSL/NNSL-KDD: 148,517 original rows and 44 columns.
- `train_test_network`: 211,043 original rows and 44 columns.

The loader discovers every top-level `.csv` file in the selected directory. It recognizes common target names such as `Label`, `Attack Type`, and `class`; handles the supplied numeric-column NSL-KDD layout; and maps benign/normal traffic to 0 and attack traffic to 1. To avoid ambiguity, retain the original target column or rename the intended target column to `label`.

Dataset filenames may vary, but names containing `cicids2017`, `nsl-kdd` or `nnsl-kdd` activate their dataset-specific safeguards. Do not place generated result CSV files in the dataset directory.

## Installation

Python 3.10 or newer is recommended. In a clean environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The exact package-version snapshot of the original runtime was not embedded in the supplied experiment archive. For strict archival reproduction, record the resolved environment after installation with `python -m pip freeze > environment-lock.txt`.

## Reproduce the experiments

First verify dataset detection and preprocessing without training:

```bash
python secure_ids_pipeline.py --data-dir /path/to/datasets --results-dir ./run/results --dry-run
```

Run the full reported configuration:

```bash
python secure_ids_pipeline.py \
  --data-dir /path/to/datasets \
  --results-dir ./run/results \
  --epochs 20 \
  --batch-size 256 \
  --seeds 42,43,44,45,46 \
  --attack-max-samples 20000 \
  --attack-epochs 10 \
  --shadow-models 2
```

For a quick functional test, use one seed, fewer epochs, and a stratified row limit:

```bash
python secure_ids_pipeline.py --data-dir /path/to/datasets --results-dir ./smoke/results --seeds 42 --epochs 1 --attack-epochs 1 --max-rows 10000
```

## Google Colab

Mount Google Drive, then run the script from a notebook cell:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
!python /content/secure_ids_pipeline.py \
  --data-dir "/content/drive/MyDrive/Secure_IDS_Reviewer_Attacks_v5/datasets" \
  --results-dir "/content/drive/MyDrive/Secure_IDS_Reviewer_Attacks_v5/results" \
  --epochs 20 --batch-size 256 --seeds 42,43,44,45,46
```

Enable a GPU in Colab through **Runtime → Change runtime type → T4 GPU** before the full run.

## Main outputs

- `results_mean.csv`, `results_std.csv`, and `results_mean_plus_std.csv`: aggregated detection, timing, overhead, throughput, memory, and reconstruction metrics.
- `all_runs_raw.csv`: per-dataset, per-seed measurements.
- `confusion_matrices.csv`: confusion-matrix counts. The corresponding `confusion_*.png` plots are in the complete ZIP archive.
- `reconstruction_attack_results.csv`: reconstruction attack evaluation.
- `membership_inference_results.csv`: shadow-model membership inference.
- `attribute_inference_results.csv`: sensitive-attribute inference.
- `ablation_study_results.csv`: five-case classification ablation.
- `ablation_reconstruction.csv`: lossless reconstruction verification for the protection pipeline.
- `dataset_audit.csv`: preprocessing, class mapping, and leakage-audit metadata.

## Pretrained weights

Pretrained model weights were not included in the supplied experiment artifacts. The script trains every model from the documented seeds and writes the reported evaluation tables. If weights are archived in a future release, their dataset, seed, preprocessing state, and checksum should be documented here.

## Reproducibility note

Deep-learning results can show small platform-dependent numerical variation because of GPU kernels, TensorFlow/CUDA versions, and hardware. The fixed seeds, group-aware split, train-only preprocessing, per-seed tables, and complete aggregated outputs are supplied to make such variation auditable.
