# KET-UNet Denoising Module (PyTorch)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](#5-environment--dependencies)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)](#5-environment--dependencies)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)](#5-environment--dependencies)
[![Status](https://img.shields.io/badge/Status-Data%20uploading%20to%20Zenodo-yellow)](#3-dataset-zenodo)

> This repository implements the **Knowledge-Embedded Transformer-UNet (KET-UNet)** denoising module for low-dose atomic-resolution STEM image enhancement.  
> **Goal:** suppress strong background/support contrast and correlated noise while preserving weak atomic peaks—enabling robust downstream atom detection and trajectory tracking.

---

## Table of Contents
- [1. Overview](#1-overview)
- [2. Repository Structure](#2-repository-structure)
- [3. Dataset (Zenodo)](#3-dataset-zenodo)
- [4. Quick Start](#4-quick-start)
- [5. Environment & Dependencies](#5-environment--dependencies)
- [6. Training (Two-stage S2RTL)](#6-training-two-stage-s2rtl)
- [7. Checkpoints & Logs](#7-checkpoints--logs)
- [8. Testing & Evaluation](#8-testing--evaluation)
- [9. I/O Details: What is Read/Written Where](#9-io-details-what-is-readwritten-where)
- [10. Reproducibility Notes](#10-reproducibility-notes)
- [11. Citation](#11-citation)
- [12. License](#12-license)
- [13. Contact](#13-contact)

---

## 1. Overview

### DAI2SY in one diagram (paper-level pipeline)

```
Raw in situ AC-STEM frames (low-dose, noisy, strong support contrast)
        │
        ▼
[KET-UNet denoising]  ← this repo (train.py / test.py)
        │
        ▼
YOLO-based single-atom detection + association tracking (e.g., ByteTrack)  ← external
        │
        ▼
Trajectory statistics (vibration vs hopping, directionality, energy landscape probing)
```

### What you get in this repository
- ✅ **KET-UNet denoising model** (`KET_Unet/KET_Unet.py`)
- ✅ **Two-stage training script** (`train.py`) implementing S2RTL-style training flow
- ✅ **Testing & metric export** (`test.py`) including image dumps and an Excel report (MSE / PSNR / MAE)
- ✅ **Dataset loader utilities** (`tools/data_pre.py`) and visualization helpers (`tools/utils.py`)

> Note: The **tracking module (YOLO + ByteTrack)** is part of the full DAI2SY framework but is not included in this repo.

---

## 2. Repository Structure

```
Github
├── .idea
├── data_save
├── KET_Unet
│   └── KET_Unet.py
├── tools
│   ├── data_pre.py
│   └── utils.py
├── train.py
└── test.py
```

### Expected dataset layout
`data_save` should contain three splits:

```
data_save/
  First_training_data/
    input/   # noisy images
    label/   # ground-truth images (same filename as input)
  Second_training_data/
    input/
    label/
  test_data/
    input/
    label/
```

**Important rule:** `input/xxx.png` must match `label/xxx.png` by filename.

---

## 3. Dataset (Zenodo)

We provide all training/testing data via **Zenodo**.

- **Zenodo DOI / link:** https://zenodo.org/records/17980551
- Suggested versioning:
  - `v1`: initial public release
  - `v1.1`: fixes/metadata update
  - `v2`: expanded real-data annotations, etc.

### Splits
- **First_training_data**: Stage-1 pretraining (typically large hybrid synthetic-to-real set)
- **Second_training_data**: Stage-2 fine-tuning (few-shot real labeled set; e.g., 4 images)
- **test_data**: held-out evaluation set

After download, unzip to repository root and ensure folder names match **exactly**.

---

## 4. Quick Start

### 4.1 Install dependencies
```bash
pip install -r requirements.txt
```
If you don’t have `requirements.txt` yet, see [Dependencies](#5-environment--dependencies).

### 4.2 Put data under `data_save/` (see Section 3)
Ensure:
```
data_save/First_training_data/input
data_save/First_training_data/label
...
```

### 4.3 Train (two-stage)
```bash
python train.py
```

### 4.4 Test a checkpoint
Edit the checkpoint path in `test.py`:

```python
test_main(args, model_file_path="model_save/--16点39分--KET_UNet--/params/_2_199_0.pth")
```

Then run:
```bash
python test.py
```

---

## 5. Environment & Dependencies

### Recommended
- Python **3.8+**
- PyTorch **1.10+**
- torchvision
- numpy, pandas
- pillow (PIL)
- openpyxl
- scikit-image

Example:
```bash
pip install torch torchvision numpy pandas pillow openpyxl scikit-image
```

> GPU is optional but strongly recommended for training.

---

## 6. Training (Two-stage S2RTL)

Training is implemented in **`train.py`** as a sequential two-stage procedure.

### Stage 1 (flag = 1): pretraining on `First_training_data`
```python
data_loader = DataLoader(
    MyDataset(os.path.join("data_save", "First_training_data")),
    batch_size=args.batch_size,
    shuffle=True
)
train_data(..., flag=1)
```

### Stage 2 (flag = 2): fine-tuning on `Second_training_data`
```python
data_loader = DataLoader(
    MyDataset(os.path.join("data_save", "Second_training_data")),
    batch_size=args.batch_size,
    shuffle=True
)
train_data(..., flag=2)
```

### Output directory convention
Training creates:

```
model_save/
  --{args.time}--{args.model_name}--/
    params/         # checkpoints .pth
    train_process/  # visualizations + train_process.xlsx
```

---

## 7. Checkpoints & Logs

### 7.1 Checkpoint saving policy

In `save_checkpoint()`:

- Stage-1 (flag=1): save every **50** batches
- Stage-2 (flag=2): save every **100** batches

Saved as:
```
model_save/--TIME--KET_UNet--/params/_FLAG_EPOCH_BATCH.pth
```

Examples:
- `_1_10_50.pth` → stage 1, epoch 10, batch 50
- `_2_199_0.pth` → stage 2, epoch 199, batch 0

### 7.2 Qualitative snapshots during training
At each checkpoint interval, `print_epoch_picture(...)` dumps images for quick inspection:

- input (noisy)
- label (target)
- output (denoised)

Saved under:
```
model_save/--TIME--KET_UNet--/train_process/
```

### 7.3 Loss export (Excel)
At the end of training:

```
model_save/--TIME--KET_UNet--/train_process/train_process.xlsx
```

> Note: `train_losses.append(train_loss.item())` is currently called twice in your script; if you intended once, you can remove one line later (not required for usage).

---

## 8. Testing & Evaluation

Testing is implemented in **`test.py`**.

### 8.1 What test.py does
1. Loads a model checkpoint (`.pth`)
2. Runs inference on `data_save/test_data`
3. Saves:
   - denoised output images
   - input images
   - concatenated visualizations: **input | label | output**
4. Computes and exports average metrics:
   - **MSE**
   - **PSNR**
   - **MAE**
   - Saves metrics to an Excel file

### 8.2 Where results are written
Given checkpoint:
```
model_save/--TIME--KET_UNet--/params/_2_199_0.pth
```

The outputs go to:
```
model_save/--TIME--KET_UNet--/
  --HH--MM--测试输出结果/
  --HH--MM--测试输入结果/
  --HH--MM--测试输入和输出结果/
  --HH--MM--误差结果.xlsx
```

### 8.3 Metrics notes
The script rescales to 0–255 before computing metrics:
```python
outputs = outputs * 255
targets = targets * 255
```

PSNR uses:
```python
skimage.metrics.peak_signal_noise_ratio
```

---

## 9. I/O Details: What is Read/Written Where

### Training (train.py)
**Reads**
- `data_save/First_training_data/input/*`
- `data_save/First_training_data/label/*`
- `data_save/Second_training_data/input/*`
- `data_save/Second_training_data/label/*`

**Writes**
- `model_save/--TIME--KET_UNet--/params/*.pth`
- `model_save/--TIME--KET_UNet--/train_process/*`
- `model_save/--TIME--KET_UNet--/train_process/train_process.xlsx`

### Testing (test.py)
**Reads**
- `data_save/test_data/input/*`
- `data_save/test_data/label/*`
- `model_save/.../params/*.pth`

**Writes**
- `model_save/.../--HH--MM--测试输出结果/*.png`
- `model_save/.../--HH--MM--测试输入结果/*.png`
- `model_save/.../--HH--MM--测试输入和输出结果/*.png`
- `model_save/.../--HH--MM--误差结果.xlsx`

---

## 10. Reproducibility Notes

- **Filename pairing is strict**: input and label filenames must match.
- If you see mismatched pairs, check:
  - wrong folder name (`label` vs `lable`)
  - different extensions (`.png` vs `.jpg`)
  - hidden suffixes

### Common pitfalls
- If `torch.load(model_file_path)` fails:
  - checkpoint path incorrect
  - PyTorch version mismatch
  - checkpoint saved on GPU, loaded on CPU (use `map_location` if needed later)

---

## 11. Citation

Please cite the DAI2SY paper if you use this work.

```bibtex
@article{DAI2SY2025,
  title   = {Deep Atomic-resolution Imaging and AI Sensing with YOLO (DAI2SY)},
  author  = {To be updated},
  journal = {To be updated},
  year    = {2025}
}
```

*(Will be updated once the preprint / publication link is available.)*

---

## 12. License
Academic research use only (license will be finalized upon public release).

---

## 13. Contact
For questions, bug reports, or collaboration:
- Maintainer: **(Your name / lab / email here)**
- Please open a GitHub Issue with:
  - OS, Python version, PyTorch version
  - checkpoint path used
  - a minimal reproduction description

---

### Acknowledgement
This repository implements the denoising component (**KET-UNet**) of the broader **DAI2SY** framework for AI-enabled atomic-scale dynamics analysis in STEM.
