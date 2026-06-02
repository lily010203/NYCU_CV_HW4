"""Visualization script for HW4 report.

Creates:
1. training_curve.png
2. model_performance_comparison.png
3. test_degraded_vs_restored.png
4. train_degraded_restored_clean.png

Put this file in the same folder as:
    hw4_promptir_final.py
    hw4_realse_dataset/
    checkpoints_dim64_p224_alltrain_ssim015_ep200_b2/
    pred_p224_ssim015_ep200_noema_tta.npz

Run:
    python visualize_hw4_results.py
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from hw4_promptir_final import PromptIR
from hw4_promptir_final import image_to_tensor
from hw4_promptir_final import pad_to_multiple
from hw4_promptir_final import forward_with_tta
from hw4_promptir_final import calculate_psnr


def parse_training_log(log_path):
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"Cannot find log file: {log_path}")

    epochs = []
    losses = []
    pattern = re.compile(r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([0-9.]+)")

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            match = pattern.search(line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    if len(epochs) == 0:
        raise RuntimeError(f"No training loss entries found in {log_path}")

    return epochs, losses


def save_training_curve(log_path, output_dir):
    epochs, losses = parse_training_log(log_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    output_path = Path(output_dir) / "training_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved training curve to {output_path}")


def save_model_performance_chart(output_dir):
    experiments = [
        "p160 + TTA",
        "p192 + TTA",
        "p192 + all-train\nSSIM 0.10",
        "p192 + SSIM 0.10\n200 epochs",
        "p192 + SSIM 0.12",
        "p224 + SSIM 0.12",
        "p224 + SSIM 0.15",
    ]
    scores = [28.92, 29.31, 29.91, 30.95, 30.98, 31.33, 31.36]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(experiments, scores)
    plt.ylabel("Kaggle Public Score")
    plt.title("Model Performance Comparison")
    plt.ylim(min(scores) - 0.5, max(scores) + 0.3)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, score in zip(bars, scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()

    output_path = Path(output_dir) / "model_performance_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved performance chart to {output_path}")


def chw_uint8_to_hwc_float(array):
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"Expected shape (3, H, W), got {array.shape}")
    array = np.transpose(array, (1, 2, 0))
    return array.astype(np.float32) / 255.0


def pil_to_float_hwc(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image).astype(np.float32) / 255.0


def save_test_restoration_visualization(data_root, pred_npz, output_dir, file_names):
    data_root = Path(data_root)
    pred_npz = Path(pred_npz)

    if not pred_npz.exists():
        raise FileNotFoundError(f"Cannot find pred npz: {pred_npz}")

    predictions = np.load(pred_npz)
    rows = len(file_names)
    cols = 2

    plt.figure(figsize=(cols * 4, rows * 3))

    for row, file_name in enumerate(file_names):
        degraded_path = data_root / "test" / "degraded" / file_name

        if not degraded_path.exists():
            print(f"Warning: skipped missing test image {degraded_path}")
            continue
        if file_name not in predictions:
            print(f"Warning: skipped missing prediction key {file_name}")
            continue

        degraded = pil_to_float_hwc(degraded_path)
        restored = chw_uint8_to_hwc_float(predictions[file_name])

        ax = plt.subplot(rows, cols, row * cols + 1)
        ax.imshow(degraded)
        ax.set_title(f"Degraded\n{file_name}")
        ax.axis("off")

        ax = plt.subplot(rows, cols, row * cols + 2)
        ax.imshow(restored)
        ax.set_title("Restored")
        ax.axis("off")

    plt.tight_layout()
    output_path = Path(output_dir) / "test_degraded_vs_restored.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved test restoration visualization to {output_path}")


def load_model(ckpt_path, dim, device, use_ema=False):
    model = PromptIR(dim=dim).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)

    if use_ema:
        ema_state = checkpoint.get("ema", None)
        if ema_state is None or not ema_state.get("enabled", False):
            print("Warning: --use_ema was set, but this checkpoint has no valid EMA weights.")
        else:
            for name, param in model.named_parameters():
                if param.requires_grad and name in ema_state["shadow"]:
                    param.data.copy_(ema_state["shadow"][name].data)

    model.eval()
    return model


def tensor_to_float_hwc(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = tensor.numpy()
    return np.transpose(array, (1, 2, 0))


@torch.no_grad()
def restore_image(model, image_path, device, tta=True):
    image = Image.open(image_path).convert("RGB")
    tensor = image_to_tensor(image).unsqueeze(0).to(device)
    padded, original_h, original_w = pad_to_multiple(tensor)
    restored = forward_with_tta(model, padded, use_tta=tta)
    restored = restored[:, :, :original_h, :original_w]
    return restored[0]


def save_train_restoration_visualization(
    data_root,
    ckpt_path,
    output_dir,
    dim,
    train_examples,
    tta=True,
    use_ema=False,
):
    data_root = Path(data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt_path, dim=dim, device=device, use_ema=use_ema)

    rows = len(train_examples)
    cols = 3
    plt.figure(figsize=(cols * 4, rows * 3))

    for row, (degraded_name, clean_name) in enumerate(train_examples):
        degraded_path = data_root / "train" / "degraded" / degraded_name
        clean_path = data_root / "train" / "clean" / clean_name

        if not degraded_path.exists() or not clean_path.exists():
            print(f"Warning: skipped missing pair {degraded_name}, {clean_name}")
            continue

        degraded = pil_to_float_hwc(degraded_path)
        clean = pil_to_float_hwc(clean_path)

        restored_tensor = restore_image(model, degraded_path, device=device, tta=tta)
        restored = tensor_to_float_hwc(restored_tensor)

        clean_tensor = image_to_tensor(Image.open(clean_path).convert("RGB")).to(device)
        psnr = calculate_psnr(restored_tensor.unsqueeze(0), clean_tensor.unsqueeze(0))

        ax = plt.subplot(rows, cols, row * cols + 1)
        ax.imshow(degraded)
        ax.set_title(f"Degraded\n{degraded_name}")
        ax.axis("off")

        ax = plt.subplot(rows, cols, row * cols + 2)
        ax.imshow(restored)
        ax.set_title(f"Restored\nPSNR: {psnr:.2f} dB")
        ax.axis("off")

        ax = plt.subplot(rows, cols, row * cols + 3)
        ax.imshow(clean)
        ax.set_title("Clean")
        ax.axis("off")

    plt.tight_layout()
    output_path = Path(output_dir) / "train_degraded_restored_clean.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved train restoration visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./hw4_realse_dataset")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints_dim64_p224_alltrain_ssim015_ep200_b2",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./checkpoints_dim64_p224_alltrain_ssim015_ep200_b2/best_model.pth",
    )
    parser.add_argument(
        "--pred_npz",
        type=str,
        default="./pred_p224_ssim015_ep200_noema_tta.npz",
    )
    parser.add_argument("--output_dir", type=str, default="./report_figures")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--no_tta", action="store_true")
    parser.add_argument("--use_ema", action="store_true")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.checkpoint_dir) / "log.txt"

    save_training_curve(log_path, output_dir)
    save_model_performance_chart(output_dir)

    test_file_names = ["0.png", "1.png", "2.png", "3.png"]
    save_test_restoration_visualization(
        data_root=args.data_root,
        pred_npz=args.pred_npz,
        output_dir=output_dir,
        file_names=test_file_names,
    )

    train_examples = [
        ("rain-1.png", "rain_clean-1.png"),
        ("rain-100.png", "rain_clean-100.png"),
        ("snow-1.png", "snow_clean-1.png"),
        ("snow-100.png", "snow_clean-100.png"),
    ]
    save_train_restoration_visualization(
        data_root=args.data_root,
        ckpt_path=args.ckpt,
        output_dir=output_dir,
        dim=args.dim,
        train_examples=train_examples,
        tta=not args.no_tta,
        use_ema=args.use_ema,
    )

    print("\nAll figures are saved in:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
