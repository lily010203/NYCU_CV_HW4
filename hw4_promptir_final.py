import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message):
        print(message)
        with self.log_path.open("a", encoding="utf-8") as file:
            file.write(str(message) + "\n")


def image_to_tensor(image):
    array = np.asarray(image).astype(np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array)


def tensor_to_uint8_chw(tensor):
    tensor = tensor.detach().cpu().clamp(0.0, 1.0)
    array = tensor.numpy()
    array = np.round(array * 255.0).astype(np.uint8)
    return array


def calculate_psnr(pred, target, eps=1e-8):
    mse = F.mse_loss(pred.clamp(0.0, 1.0), target.clamp(0.0, 1.0))
    if mse.item() < eps:
        return 100.0
    return 20.0 * torch.log10(
        torch.tensor(1.0, device=pred.device) / torch.sqrt(mse)
    )


def pad_to_multiple(x, multiple=4):
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple

    if pad_h == 0 and pad_w == 0:
        return x, height, width

    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, height, width


class RestorationDataset(Dataset):
    def __init__(self, data_root, split="train", val_ratio=0.1, patch_size=128, seed=42):
        self.data_root = Path(data_root)
        self.split = split
        self.patch_size = patch_size

        degraded_dir = self.data_root / "train" / "degraded"
        clean_dir = self.data_root / "train" / "clean"

        all_pairs = []
        for index in range(1, 1601):
            degraded_path = degraded_dir / f"rain-{index}.png"
            clean_path = clean_dir / f"rain_clean-{index}.png"
            if degraded_path.exists() and clean_path.exists():
                all_pairs.append((degraded_path, clean_path))

        for index in range(1, 1601):
            degraded_path = degraded_dir / f"snow-{index}.png"
            clean_path = clean_dir / f"snow_clean-{index}.png"
            if degraded_path.exists() and clean_path.exists():
                all_pairs.append((degraded_path, clean_path))

        rng = random.Random(seed)
        rng.shuffle(all_pairs)
        val_size = int(len(all_pairs) * val_ratio)

        if split == "train":
            self.pairs = all_pairs if val_size == 0 else all_pairs[val_size:]
        elif split == "val":
            self.pairs = all_pairs[:val_size]
        else:
            raise ValueError("split must be 'train' or 'val'.")

        if len(self.pairs) == 0:
            raise RuntimeError("No paired images found. Please check your dataset path.")

    def __len__(self):
        return len(self.pairs)

    def _random_crop_pair(self, degraded, clean):
        _, height, width = degraded.shape
        patch_size = self.patch_size

        if height < patch_size or width < patch_size:
            pad_h = max(0, patch_size - height)
            pad_w = max(0, patch_size - width)
            degraded = F.pad(
                degraded.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
            ).squeeze(0)
            clean = F.pad(
                clean.unsqueeze(0), (0, pad_w, 0, pad_h), mode="reflect"
            ).squeeze(0)
            _, height, width = degraded.shape

        top = random.randint(0, height - patch_size)
        left = random.randint(0, width - patch_size)

        degraded = degraded[:, top:top + patch_size, left:left + patch_size]
        clean = clean[:, top:top + patch_size, left:left + patch_size]
        return degraded, clean

    @staticmethod
    def _augment_pair(degraded, clean):
        if random.random() < 0.5:
            degraded = torch.flip(degraded, dims=[2])
            clean = torch.flip(clean, dims=[2])

        if random.random() < 0.5:
            degraded = torch.flip(degraded, dims=[1])
            clean = torch.flip(clean, dims=[1])

        rotation_k = random.randint(0, 3)
        if rotation_k > 0:
            degraded = torch.rot90(degraded, rotation_k, dims=[1, 2])
            clean = torch.rot90(clean, rotation_k, dims=[1, 2])

        return degraded, clean

    def __getitem__(self, index):
        degraded_path, clean_path = self.pairs[index]

        degraded = Image.open(degraded_path).convert("RGB")
        clean = Image.open(clean_path).convert("RGB")

        degraded = image_to_tensor(degraded)
        clean = image_to_tensor(clean)

        if self.split == "train":
            degraded, clean = self._random_crop_pair(degraded, clean)
            degraded, clean = self._augment_pair(degraded, clean)

        return degraded, clean


class TestDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.degraded_dir = self.data_root / "test" / "degraded"
        self.files = sorted(self.degraded_dir.glob("*.png"), key=lambda path: int(path.stem))

        if len(self.files) == 0:
            raise RuntimeError("No test images found. Please check test/degraded.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        tensor = image_to_tensor(image)
        return path.name, tensor


class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + 1e-5)
        return x * self.weight + self.bias


class FeedForward(nn.Module):
    def __init__(self, channels, expansion=2.66):
        super().__init__()
        hidden_channels = int(channels * expansion)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MultiDconvHeadAttention(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
        )
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        query, key, value = qkv.chunk(3, dim=1)

        query = query.reshape(batch_size, self.heads, channels // self.heads, height * width)
        key = key.reshape(batch_size, self.heads, channels // self.heads, height * width)
        value = value.reshape(batch_size, self.heads, channels // self.heads, height * width)

        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = attention * self.temperature
        attention = attention.softmax(dim=-1)

        out = torch.matmul(attention, value)
        out = out.reshape(batch_size, channels, height, width)
        out = self.project_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, channels, heads=4):
        super().__init__()
        self.norm1 = LayerNorm2d(channels)
        self.attention = MultiDconvHeadAttention(channels, heads=heads)
        self.norm2 = LayerNorm2d(channels)
        self.feed_forward = FeedForward(channels)

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.feed_forward(self.norm2(x))
        return x


class PromptGenBlock(nn.Module):
    def __init__(self, channels, prompt_len=5, prompt_size=16):
        super().__init__()
        self.prompt_len = prompt_len
        self.prompt_param = nn.Parameter(
            torch.randn(1, prompt_len, channels, prompt_size, prompt_size)
        )
        self.linear = nn.Linear(channels, prompt_len)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        global_feature = x.mean(dim=(2, 3))
        prompt_weight = self.linear(global_feature)
        prompt_weight = F.softmax(prompt_weight, dim=1)

        prompt = self.prompt_param.repeat(batch_size, 1, 1, 1, 1)
        prompt_weight = prompt_weight.view(batch_size, self.prompt_len, 1, 1, 1)
        prompt = (prompt * prompt_weight).sum(dim=1)

        prompt = F.interpolate(
            prompt,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        prompt = self.conv(prompt)
        return x + prompt


class PromptIR(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        self.encoder1 = nn.Sequential(
            TransformerBlock(dim, heads=1),
            TransformerBlock(dim, heads=1),
        )
        self.down1 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)

        self.encoder2 = nn.Sequential(
            TransformerBlock(dim * 2, heads=2),
            TransformerBlock(dim * 2, heads=2),
        )
        self.down2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)

        self.latent = nn.Sequential(
            TransformerBlock(dim * 4, heads=4),
            TransformerBlock(dim * 4, heads=4),
            TransformerBlock(dim * 4, heads=4),
            TransformerBlock(dim * 4, heads=4),
        )

        self.prompt_latent = PromptGenBlock(dim * 4)
        self.prompt_dec2 = PromptGenBlock(dim * 2)
        self.prompt_dec1 = PromptGenBlock(dim)

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.decoder2 = nn.Sequential(
            TransformerBlock(dim * 2, heads=2),
            TransformerBlock(dim * 2, heads=2),
        )

        self.up1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder1 = nn.Sequential(
            TransformerBlock(dim, heads=1),
            TransformerBlock(dim, heads=1),
        )

        self.output = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x1 = self.patch_embed(x)
        enc1 = self.encoder1(x1)

        x2 = self.down1(enc1)
        enc2 = self.encoder2(x2)

        x3 = self.down2(enc2)
        latent = self.latent(x3)
        latent = self.prompt_latent(latent)

        dec2 = self.up2(latent)
        if dec2.shape[-2:] != enc2.shape[-2:]:
            dec2 = F.interpolate(dec2, size=enc2.shape[-2:], mode="bilinear", align_corners=False)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.reduce2(dec2)
        dec2 = self.decoder2(dec2)
        dec2 = self.prompt_dec2(dec2)

        dec1 = self.up1(dec2)
        if dec1.shape[-2:] != enc1.shape[-2:]:
            dec1 = F.interpolate(dec1, size=enc1.shape[-2:], mode="bilinear", align_corners=False)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.reduce1(dec1)
        dec1 = self.decoder1(dec1)
        dec1 = self.prompt_dec1(dec1)

        out = self.output(dec1)
        out = out + residual
        return out.clamp(0.0, 1.0)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def gradient_loss(pred, target):
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)


def ssim_map(pred, target):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(target, kernel_size=3, stride=1, padding=1)

    sigma_x = F.avg_pool2d(pred * pred, kernel_size=3, stride=1, padding=1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, kernel_size=3, stride=1, padding=1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, kernel_size=3, stride=1, padding=1) - mu_x * mu_y

    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return numerator / (denominator + 1e-8)


def ssim_loss(pred, target):
    ssim_value = ssim_map(pred, target).mean()
    return torch.clamp((1.0 - ssim_value) / 2.0, min=0.0, max=1.0)


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.enabled = decay > 0.0
        self.shadow = {}
        self.backup = {}

        if self.enabled:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.shadow[name] = param.detach().clone()

    @torch.no_grad()
    def update(self, model):
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        if not self.enabled:
            return
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].data)

    def restore(self, model):
        if not self.enabled:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "enabled": self.enabled, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict.get("decay", self.decay)
        self.enabled = state_dict.get("enabled", self.enabled)
        self.shadow = state_dict.get("shadow", {})


def compute_total_loss(restored, clean, args):
    loss_pixel = CharbonnierLoss()(restored, clean)
    loss_edge = gradient_loss(restored, clean)
    loss_ssim = ssim_loss(restored, clean)
    total = loss_pixel + args.edge_weight * loss_edge + args.ssim_weight * loss_ssim
    return total


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, args):
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Train Epoch {epoch}")

    for degraded, clean in progress:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            restored = model(degraded)
            loss = compute_total_loss(restored, clean, args)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.6f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, epoch, args):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    progress = tqdm(loader, desc=f"Valid Epoch {epoch}")

    for degraded, clean in progress:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        degraded, original_h, original_w = pad_to_multiple(degraded)
        restored = model(degraded)
        restored = restored[:, :, :original_h, :original_w]

        loss = compute_total_loss(restored, clean, args)
        psnr = calculate_psnr(restored, clean)

        total_loss += loss.item()
        total_psnr += psnr.item()
        progress.set_postfix(psnr=f"{psnr.item():.4f}")

    return total_loss / len(loader), total_psnr / len(loader)


def save_checkpoint(path, model, ema, optimizer, scheduler, epoch, best_metric, monitor_name, monitor_mode, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict() if ema is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "best_metric": best_metric,
            "monitor_name": monitor_name,
            "monitor_mode": monitor_mode,
            "args": vars(args),
        },
        path,
    )


def load_resume_checkpoint(resume_path, model, ema, optimizer, scheduler, device, logger):
    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)

    if ema is not None and checkpoint.get("ema") is not None:
        ema.load_state_dict(checkpoint["ema"])
    if checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])

    start_epoch = checkpoint.get("epoch", 0) + 1
    best_metric = checkpoint.get("best_metric", None)
    monitor_name = checkpoint.get("monitor_name", "val_psnr")
    monitor_mode = checkpoint.get("monitor_mode", "max")

    logger.log(f"Resumed from: {resume_path}")
    logger.log(f"Resume start epoch: {start_epoch}")
    logger.log(f"Loaded best metric ({monitor_name}): {best_metric}")

    return start_epoch, best_metric, monitor_name, monitor_mode


def is_better(current, best, mode):
    if best is None:
        return True
    return current > best if mode == "max" else current < best


def train(args):
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(checkpoint_dir / "log.txt")
    logger.log("========== TRAIN START ==========")
    logger.log("Final version note: EMA is stored in checkpoints but inference uses normal weights unless --use_ema is set.")
    logger.log(f"Args: {vars(args)}")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    train_dataset = RestorationDataset(
        data_root=args.data_root,
        split="train",
        val_ratio=args.val_ratio,
        patch_size=args.patch_size,
        seed=args.seed,
    )

    has_validation = args.val_ratio > 0.0
    val_dataset = None
    if has_validation:
        val_dataset = RestorationDataset(
            data_root=args.data_root,
            split="val",
            val_ratio=args.val_ratio,
            patch_size=args.patch_size,
            seed=args.seed,
        )
        if len(val_dataset) == 0:
            has_validation = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    val_loader = None
    if has_validation:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        logger.log(f"Train pairs: {len(train_dataset)} | Val pairs: {len(val_dataset)}")
    else:
        logger.log(f"Train pairs: {len(train_dataset)} | Val disabled (val_ratio={args.val_ratio})")

    model = PromptIR(dim=args.dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler(device="cuda", enabled=device.type == "cuda")
    ema = ModelEMA(model, decay=args.ema_decay)

    monitor_name = "val_psnr" if has_validation else "train_loss"
    monitor_mode = "max" if has_validation else "min"
    best_metric = None
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_metric, monitor_name, monitor_mode = load_resume_checkpoint(
            resume_path, model, ema, optimizer, scheduler, device, logger
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        if ema.enabled:
            ema.update(model)

        current_metric = train_loss
        if has_validation:
            ema.apply_to(model)
            val_loss, val_psnr = validate(model, val_loader, device, epoch, args)
            ema.restore(model)
            current_metric = val_psnr
            logger.log(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val PSNR: {val_psnr:.4f}"
            )
        else:
            logger.log(f"Epoch {epoch:03d} | Train Loss: {train_loss:.6f}")

        save_checkpoint(
            checkpoint_dir / "last_model.pth",
            model,
            ema,
            optimizer,
            scheduler,
            epoch,
            best_metric,
            monitor_name,
            monitor_mode,
            args,
        )

        if is_better(current_metric, best_metric, monitor_mode):
            best_metric = current_metric
            save_checkpoint(
                checkpoint_dir / "best_model.pth",
                model,
                ema,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                monitor_name,
                monitor_mode,
                args,
            )
            if monitor_mode == "max":
                logger.log(f"Saved best model. Best {monitor_name}: {best_metric:.4f}")
            else:
                logger.log(f"Saved best model. Best {monitor_name}: {best_metric:.6f}")

        scheduler.step()

    logger.log("========== TRAIN END ==========")


def apply_tta_transform(x, mode):
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=[3])
    if mode == 2:
        return torch.flip(x, dims=[2])
    if mode == 3:
        return torch.flip(x, dims=[2, 3])
    if mode == 4:
        return torch.rot90(x, k=1, dims=[2, 3])
    if mode == 5:
        return torch.rot90(x, k=2, dims=[2, 3])
    if mode == 6:
        return torch.rot90(x, k=3, dims=[2, 3])
    if mode == 7:
        return torch.flip(torch.rot90(x, k=1, dims=[2, 3]), dims=[3])
    raise ValueError(f"Unsupported TTA mode: {mode}")


def invert_tta_transform(x, mode):
    if mode == 0:
        return x
    if mode == 1:
        return torch.flip(x, dims=[3])
    if mode == 2:
        return torch.flip(x, dims=[2])
    if mode == 3:
        return torch.flip(x, dims=[2, 3])
    if mode == 4:
        return torch.rot90(x, k=3, dims=[2, 3])
    if mode == 5:
        return torch.rot90(x, k=2, dims=[2, 3])
    if mode == 6:
        return torch.rot90(x, k=1, dims=[2, 3])
    if mode == 7:
        return torch.rot90(torch.flip(x, dims=[3]), k=3, dims=[2, 3])
    raise ValueError(f"Unsupported TTA mode: {mode}")


@torch.no_grad()
def forward_with_tta(model, image, use_tta=False):
    if not use_tta:
        return model(image)

    outputs = []
    for mode in range(8):
        transformed = apply_tta_transform(image, mode)
        restored = model(transformed)
        restored = invert_tta_transform(restored, mode)
        outputs.append(restored)

    return torch.stack(outputs, dim=0).mean(dim=0)


@torch.no_grad()
def infer(args):
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger = Logger(output_path.parent / "infer_log.txt")
    logger.log("========== INFER START ==========")
    logger.log(f"Args: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Using device: {device}")

    model = PromptIR(dim=args.dim).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)

    ema_state = checkpoint.get("ema", None)
    if args.use_ema and ema_state is not None and ema_state.get("enabled", False):
        logger.log("Applying EMA weights for inference because --use_ema was set.")
        for name, param in model.named_parameters():
            if param.requires_grad and name in ema_state["shadow"]:
                param.data.copy_(ema_state["shadow"][name].data)
    elif args.use_ema:
        logger.log("Warning: --use_ema was set, but this checkpoint has no EMA weights.")
    else:
        logger.log("Using normal model weights for inference. EMA is disabled by default.")

    model.eval()
    test_dataset = TestDataset(args.data_root)
    restored_dict = {}

    for filename, image in tqdm(test_dataset, desc="Inference"):
        image = image.unsqueeze(0).to(device)
        padded, original_h, original_w = pad_to_multiple(image)
        restored = forward_with_tta(model, padded, use_tta=args.tta)
        restored = restored[:, :, :original_h, :original_w]
        restored_dict[filename] = tensor_to_uint8_chw(restored[0])

    np.savez(args.output, **restored_dict)
    logger.log(f"Saved {len(restored_dict)} images to {args.output}")
    logger.log("========== INFER END ==========")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", type=str, choices=["train", "infer"], required=True)
    parser.add_argument("--data_root", type=str, default="./hw4_realse_dataset")

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--patch_size", type=int, default=224)
    parser.add_argument("--val_ratio", type=float, default=0.0)

    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--edge_weight", type=float, default=0.05)
    parser.add_argument("--ssim_weight", type=float, default=0.15)
    parser.add_argument("--ema_decay", type=float, default=0.999)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--ckpt", type=str, default="./checkpoints/best_model.pth")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--output", type=str, default="./pred.npz")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Use EMA weights during inference. Disabled by default.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    elif args.mode == "infer":
        infer(args)


if __name__ == "__main__":
    main()
