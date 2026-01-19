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
  - [4.4 Online Demo (Coze Bot)](#44-online-demo-coze-bot)
  - [4.5 Detection & Tracking (YOLO11 + ByteTrack, External)](#45-detection--tracking-yolo11--bytetrack-external)
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
YOLO11-based single-atom detection + association tracking (e.g., ByteTrack)  ← external
        │
        ▼
Trajectory statistics (vibration vs hopping, directionality, energy landscape probing)
```

### What you get in this repository
- ✅ **KET-UNet denoising model** (`KET_Unet/KET_Unet.py`)
- ✅ **Two-stage training script** (`train.py`) implementing S2RTL-style training flow
- ✅ **Testing & metric export** (`test.py`) including image dumps and an Excel report (MSE / PSNR / MAE)
- ✅ **Dataset loader utilities** (`tools/data_pre.py`) and visualization helpers (`tools/utils.py`)

> Note: The full DAI2SY framework includes a **detection + tracking** module that calls **Ultralytics YOLO11** and a standard association tracker (e.g., ByteTrack).  
> We do **not** duplicate YOLO source code in this repo. Instead, we provide minimal, reproducible wrapper code and commands for calling upstream YOLO11.

---

## Tracking Module (External): YOLO11 + ByteTrack

### Why YOLO code is not bundled here
Ultralytics YOLO is distributed under the **AGPL-3.0 license by default**. To avoid duplicating upstream code and to keep this repository focused on denoising, we do not copy YOLO internals into this repo.  
Instead, we provide **minimal wrapper scripts** that call the official Ultralytics implementation. This is sufficient for reproducing the detection/tracking stage used in the full pipeline.

### What you should cite / acknowledge
If you use the detection/tracking stage, please acknowledge:
- **Ultralytics YOLO11** (upstream implementation, models, and CLI/Python API)
- **ByteTrack** (or your chosen association tracker)

### Minimal wrapper code (core call logic)
The following scripts are designed to be copy-pasted into this repo under a new folder `tracking/`. They demonstrate exactly how YOLO11 is called for:
1) detection on denoised frames and
2) tracking on a video/stream.

#### `tracking/yolo11_detect_images.py`
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run YOLO11 detection on a folder of images (e.g., denoised STEM frames).
Outputs: annotated images + prediction TXT (Ultralytics default structure).
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="yolo11n.pt",
                   help="YOLO11 weights path (e.g., yolo11n.pt or weights/yolo11_atom.pt)")
    p.add_argument("--source", type=str, required=True,
                   help="Image folder or a glob pattern, e.g., outputs_denoised/*.png")
    p.add_argument("--out", type=str, default="runs/yolo11_detect",
                   help="Output directory root (Ultralytics will create subfolders)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default="0",
                   help="0 / 0,1 / cpu")
    return p.parse_args()

def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.out,
        name="pred",
        save=True,       # save annotated images
        save_txt=True,   # save labels in YOLO format
        save_conf=True
    )

if __name__ == "__main__":
    main()
```

#### `tracking/yolo11_track_video.py`
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run YOLO11 tracking (ByteTrack) on a video or stream.
This uses Ultralytics built-in track mode with tracker config (bytetrack.yaml).
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--source", type=str, required=True,
                   help="Video path or stream (e.g., video.mp4). For webcam use source=0")
    p.add_argument("--tracker", type=str, default="bytetrack.yaml",
                   help="Ultralytics tracker config (bytetrack.yaml or botsort.yaml)")
    p.add_argument("--out", type=str, default="runs/yolo11_track")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()

def main():
    args = parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    model.track(
        source=args.source,
        tracker=args.tracker,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.out,
        name="track",
        save=True,      # save annotated video
        save_txt=True,  # save per-frame tracks to txt
        persist=True
    )

if __name__ == "__main__":
    main()
```

> If you trained a custom atom detector, replace `yolo11n.pt` with your checkpoint path (e.g., `weights/yolo11_atom.pt`).  
> If you prefer a different association tracker, you may swap `bytetrack.yaml` accordingly.

---

## 2. Repository Structure

```
├── data_save
├── KET_Unet
│   └── KET_Unet.py
├── tools
│   ├── data_pre.py
│   └── utils.py
├── tracking              # (recommended) external detection/tracking wrappers
│   ├── yolo11_detect_images.py
│   └── yolo11_track_video.py
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
test_main(args, model_file_path)
```

Then run:
```bash
python test.py
```

---

### 4.4 Online Demo (Coze Bot)

**Note:** This model has been deployed as an AI agent on the Coze platform, allowing users to perform denoising through conversational interaction without local installation.

- **Agent Link:** [https://www.coze.cn/s/6rr38GMmSzI/](https://www.coze.cn/s/6rr38GMmSzI/)

### Usage Instructions
1. Open the link above to access the Coze agent
2. Upload the image you want to denoise (supports common image formats)
3. The agent will return the denoised image result

### Important Notes
- Due to server computing limitations, **consecutive rapid uploads of multiple images may result in 500 errors**
- If this occurs, please **delete the current conversation history** and start a new upload
- We recommend focusing on single-image denoising per conversation session
- For the data images in this demo, you can choose any image from Zenodo DOI / link: https://zenodo.org/records/17980551.

---

### 4.5 Detection & Tracking (YOLO11 + ByteTrack, External)

This repository focuses on denoising. For detection/tracking in the full DAI2SY pipeline, we call **Ultralytics YOLO11** and (optionally) **ByteTrack** for association.

#### 4.5.1 Install external tracking dependency
```bash
pip install -U ultralytics
```

#### 4.5.2 Detect atoms on denoised frames (image folder)
```bash
python tracking/yolo11_detect_images.py   --source "model_save/--TIME--KET_UNet--/--HH--MM--/*.png"   --weights yolo11n.pt   --conf 0.25 --imgsz 640
```

#### 4.5.3 Track across frames (video/stream)
```bash
python tracking/yolo11_track_video.py   --source path/to/video.mp4   --weights yolo11n.pt   --tracker bytetrack.yaml
```

> For custom atom weights, replace `yolo11n.pt` with `weights/yolo11_atom.pt`.

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

Optional (for external detection/tracking):
- ultralytics

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
- `model_save/.../--HH--MM--/*.png`
- `model_save/.../--HH--MM--/*.png`
- `model_save/.../--HH--MM--/*.png`
- `model_save/.../--HH--MM--.xlsx`

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

**Note on external dependencies:** Ultralytics YOLO is licensed under **AGPL-3.0 by default**; if you use YOLO11 in your pipeline, please comply with the upstream license.

---

## 13. Contact
For questions, bug reports, or collaboration:
- Please open a GitHub Issue with:
  - OS, Python version, PyTorch version
  - checkpoint path used
  - a minimal reproduction description

---

### Acknowledgement
This repository implements the denoising component (**KET-UNet**) of the broader **DAI2SY** framework for AI-enabled atomic-scale dynamics analysis in STEM.
