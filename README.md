# NYCU Computer Vision 2026 HW4

- **Student ID:** 313554063
- **Task:** Image Restoration for Rain and Snow Degradation
- **Final Public Score:** 31.36

## Introduction

This project implements a single image restoration model for two degradation types: rain and snow.  
The goal is to restore degraded RGB images into clean images and generate the required `pred.npz` submission file.

The final model is a compact PromptIR-style restoration network. It uses an encoder-decoder transformer architecture with prompt generation blocks, residual learning, Charbonnier loss, edge loss, SSIM loss, and test-time augmentation.

The final submission was generated using:

- `dim = 64`
- `patch_size = 224`
- `batch_size = 2`
- `epochs = 200`
- `ssim_weight = 0.15`
- `val_ratio = 0.0`
- TTA inference enabled
- EMA inference disabled

## Environment Setup

The code was tested with Python 3.12 and PyTorch.

Install the required packages:

```bash
pip install numpy pillow tqdm matplotlib torch torchvision
```

If you use a virtual environment, activate it first:

```bash
source /home/user/myenv/bin/activate
```

Then install the dependencies:

```bash
python -m pip install numpy pillow tqdm matplotlib torch torchvision
```

## Dataset Structure

Please place the dataset in the following structure:

```text
.
├── hw4_promptir_final.py
├── visualize_hw4_results.py
├── hw4_realse_dataset/
│   ├── train/
│   │   ├── degraded/
│   │   │   ├── rain-1.png ... rain-1600.png
│   │   │   └── snow-1.png ... snow-1600.png
│   │   └── clean/
│   │       ├── rain_clean-1.png ... rain_clean-1600.png
│   │       └── snow_clean-1.png ... snow_clean-1600.png
│   └── test/
│       └── degraded/
│           ├── 0.png ... 99.png
```

## Usage

### Training

To reproduce the final training setting:

```bash
python hw4_promptir_final.py --mode train \
  --data_root ./hw4_realse_dataset \
  --epochs 200 \
  --dim 64 \
  --patch_size 224 \
  --batch_size 2 \
  --val_ratio 0.0 \
  --ssim_weight 0.15 \
  --checkpoint_dir ./checkpoints_final
```

The checkpoints and training log will be saved to:

```text
./checkpoints_final/
├── best_model.pth
├── last_model.pth
└── log.txt
```

Since `val_ratio = 0.0`, all 3200 training image pairs are used for training.  
In this mode, `best_model.pth` is selected according to the lowest training loss.

### Resume Training

To resume from a previous checkpoint:

```bash
python hw4_promptir_final.py --mode train \
  --data_root ./hw4_realse_dataset \
  --epochs 200 \
  --dim 64 \
  --patch_size 224 \
  --batch_size 2 \
  --val_ratio 0.0 \
  --ssim_weight 0.15 \
  --resume ./checkpoints_final/last_model.pth \
  --checkpoint_dir ./checkpoints_final_resume
```

### Inference

To generate the final `pred.npz` file:

```bash
python hw4_promptir_final.py --mode infer \
  --data_root ./hw4_realse_dataset \
  --dim 64 \
  --ckpt ./checkpoints_final/best_model.pth \
  --tta \
  --output ./pred.npz
```

The output file will be:

```text
pred.npz
```

The submission file stores each restored image as a NumPy array in `(3, H, W)` format with `uint8` values.

### EMA Note

EMA weights are saved in the checkpoint for experimental purposes, but the final submission does **not** use EMA inference.

By default, inference uses normal model weights.  
Do not add `--use_ema` for the final submission.

If you want to test EMA inference manually:

```bash
python hw4_promptir_final.py --mode infer \
  --data_root ./hw4_realse_dataset \
  --dim 64 \
  --ckpt ./checkpoints_final/best_model.pth \
  --tta \
  --use_ema \
  --output ./pred_ema.npz
```

## Visualization

To generate figures for the report:

```bash
python visualize_hw4_results.py \
  --data_root ./hw4_realse_dataset \
  --checkpoint_dir ./checkpoints_final \
  --ckpt ./checkpoints_final/best_model.pth \
  --pred_npz ./pred.npz \
  --output_dir ./report_figures
```

The script will generate:

```text
report_figures/
├── training_curve.png
├── model_performance_comparison.png
├── test_degraded_vs_restored.png
└── train_degraded_restored_clean.png
```

These figures include:

- training loss curve
- public score comparison across experiments
- test degraded/restored visualization
- training degraded/restored/clean visualization with PSNR

## Method Summary

### Model Architecture

The model is a compact PromptIR-style restoration network with:

- convolutional patch embedding
- encoder-decoder transformer structure
- multi-head depth-wise convolution attention
- gated feed-forward network
- prompt generation blocks
- skip connections
- residual image prediction

### Loss Function

The final loss function is:

```text
Total Loss = Charbonnier Loss + 0.05 × Edge Loss + 0.15 × SSIM Loss
```

Charbonnier loss is used for pixel-level reconstruction.  
Edge loss helps preserve image structures and boundaries.  
SSIM loss improves structural similarity between the restored image and the clean target.

### Test-Time Augmentation

TTA is used during inference.  
Each test image is transformed using flips and rotations. The model restores all transformed versions, and the final output is obtained by averaging the inverse-transformed predictions.

## Performance Snapshot

| Experiment | Public Score |
|---|---:|
| patch160 + TTA | 28.92 |
| patch192 + TTA | 29.31 |
| patch192 + all-train + SSIM 0.10 | 29.91 |
| patch192 + SSIM 0.10 + 200 epochs | 30.95 |
| patch192 + SSIM 0.12 + 200 epochs | 30.98 |
| patch224 + SSIM 0.12 + 200 epochs | 31.33 |
| patch224 + SSIM 0.15 + 200 epochs | **31.36** |

The best public score is **31.36**, achieved by the final setting:

```text
dim = 64
patch_size = 224
batch_size = 2
epochs = 200
ssim_weight = 0.15
TTA = enabled
EMA inference = disabled
```

## File Description

```text
hw4_promptir_final.py
```

Main training and inference script.

```text
visualize_hw4_results.py
```

Script for generating report figures.

```text
pred.npz
```

Final prediction file for Kaggle submission.

```text
checkpoints_final/best_model.pth
```

Best checkpoint selected by training loss in the final all-train setting.

## References

1. V. Potlapalli, S. W. Zamir, S. Khan, and F. S. Khan, "PromptIR: Prompting for All-in-One Blind Image Restoration," Advances in Neural Information Processing Systems, 2023.
2. S. W. Zamir, A. Arora, S. Khan, M. Hayat, F. S. Khan, M.-H. Yang, and L. Shao, "Restormer: Efficient Transformer for High-Resolution Image Restoration," CVPR, 2022.
