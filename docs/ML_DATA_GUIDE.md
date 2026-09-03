# OceanEmbed ML Team Quickstart Guide
**SIH Problem Statement #26066 — 3D Ocean Temperature Reconstruction**

---

## 1. Processed Dataset Files (`data/processed/`)

| File Name | Shape / Dimensions | Dtype | Description |
|---|---|---|---|
| `train_inputs.npy` | `(N_train, 7, 101, 241)` | `float32` | Z-score normalized 7 surface input channels |
| `train_targets.npy` | `(N_train, 15, 101, 241)` | `float32` | Subsurface temperature (°C) at 15 depth levels |
| `val_inputs.npy` | `(N_val, 7, 101, 241)` | `float32` | Validation input tensor |
| `val_targets.npy` | `(N_val, 15, 101, 241)` | `float32` | Validation target tensor |
| `test_inputs.npy` | `(N_test, 7, 101, 241)` | `float32` | Test input tensor (held-out) |
| `test_targets.npy` | `(N_test, 15, 101, 241)` | `float32` | Test target tensor |
| `ocean_mask.npy` | `(101, 241)` | `bool` | `True` = Ocean (11,854 px), `False` = Land |
| `norm_stats.json` | Key-Value dict | `json` | Mean ($\mu$) and Std ($\sigma$) computed on Train split |
| `split_info.json` | Metadata dict | `json` | Exact calendar dates and timestamps per split |

---

## 2. Channel & Depth Mappings

### 7 Input Channels ($C = 7$)
```python
Channel 0: SST     # Sea Surface Temperature (°C, z-score normalized)
Channel 1: SSS     # Sea Surface Salinity (PSU, z-score normalized)
Channel 2: SSH     # Sea Surface Height / Sea Level Anomaly (m, z-score normalized)
Channel 3: U_curr  # Eastward Surface Current (m/s, z-score normalized)
Channel 4: V_curr  # Northward Surface Current (m/s, z-score normalized)
Channel 5: U_wind  # Eastward 10m Wind (m/s, z-score normalized)
Channel 6: V_wind  # Northward 10m Wind (m/s, z-score normalized)
```

### 15 Target Depth Levels ($D = 15$)
```python
TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000] # meters
```
* **Units**: Targets are in **actual physical temperature (°C)**.

---

## 3. Spatial Domain & Grid
* **Latitude**: $5.0^\circ\text{N} \to 30.0^\circ\text{N}$ ($101$ points, step $= 0.25^\circ$)
* **Longitude**: $45.0^\circ\text{E} \to 105.0^\circ\text{E}$ ($241$ points, step $= 0.25^\circ$)

---

## 4. ⚠️ Critical Rules for Model Training

### A. Loss Masking (MANDATORY)
Land pixels are zero-padded (`0.0`). You **MUST mask out land pixels** when computing loss functions (MSE, MAE, Huber) to avoid penalizing the model for land zeros:

```python
import torch

class MaskedMSELoss(torch.nn.Module):
    def __init__(self, mask_tensor):
        super().__init__()
        # mask shape: (1, 1, 101, 241) or (101, 241)
        self.register_buffer("mask", mask_tensor.float())

    def forward(self, y_pred, y_true):
        # y_pred, y_true: (B, 15, 101, 241)
        diff_sq = (y_pred - y_true) ** 2
        masked_diff = diff_sq * self.mask
        loss = masked_diff.sum() / (self.mask.sum() * y_pred.shape[1] * y_pred.shape[0])
        return loss
```

### B. Input De-normalization
To convert normalized inputs $z$ back to physical units:
$$x_{\text{physical}} = (z \times \sigma) + \mu$$

---

## 5. Ready-to-Use PyTorch DataLoader

```python
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class OceanDataset(Dataset):
    def __init__(self, data_dir="data/processed", split="train"):
        data_path = Path(data_dir)
        self.inputs = torch.from_numpy(np.load(data_path / f"{split}_inputs.npy"))
        self.targets = torch.from_numpy(np.load(data_path / f"{split}_targets.npy"))
        self.mask = torch.from_numpy(np.load(data_path / "ocean_mask.npy"))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# Usage
train_ds = OceanDataset(split="train")
train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

for x, y in train_loader:
    print(f"Batch X shape: {x.shape}")  # torch.Size([4, 7, 101, 241])
    print(f"Batch Y shape: {y.shape}")  # torch.Size([4, 15, 101, 241])
    break
```
