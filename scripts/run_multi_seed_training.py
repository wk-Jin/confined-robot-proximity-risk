"""
Run final multi-seed training for image-based risk regression models.

This script trains ResNet-18 and ViT-Tiny with the selected hyperparameters
over multiple training seeds. The layout split seed is kept fixed while the
training seed changes, matching the experimental protocol used for reporting
mean and standard deviation across seeds.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from pathlib import Path


np = None
pd = None
Image = None
torch = None
nn = None
optim = None
transforms = None
models = None
timm = None


DEFAULT_HYPERPARAMETERS = {
    "resnet18": {"learning_rate": 1e-4, "weight_decay": 1e-6},
    "vit_tiny": {"learning_rate": 1e-4, "weight_decay": 1e-5},
}


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one integer seed.")
    return values


def load_libraries(require_timm: bool = False) -> None:
    """Import heavy ML libraries after CLI parsing so --help stays lightweight."""
    global np, pd, Image, torch, nn, optim, transforms, models, timm

    try:
        import numpy as np_module
        import pandas as pd_module
        from PIL import Image as image_module
        import torch as torch_module
        from torch import nn as nn_module
        from torch import optim as optim_module
        from torchvision import models as models_module
        from torchvision import transforms as transforms_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install the project requirements "
            "before running final training."
        ) from exc

    np = np_module
    pd = pd_module
    Image = image_module
    torch = torch_module
    nn = nn_module
    optim = optim_module
    transforms = transforms_module
    models = models_module

    if require_timm:
        try:
            import timm as timm_module
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: timm. Install timm to train ViT-Tiny.") from exc
        timm = timm_module


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=json_default)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_success_mask(series):
    if series.dtype == bool:
        return series

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int) == 1

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["1", "true", "yes", "y"])


def resolve_image_path(raw_value, image_dir: Path) -> Path:
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text.lower() == "nan":
        raise FileNotFoundError("Empty image path in CSV.")

    normalized = raw_text.replace("\\", "/")
    raw_path = Path(normalized)
    candidates: list[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(image_dir.parent / raw_path)

    candidates.append(image_dir / raw_path.name)

    if raw_path.suffix.lower() == ".png":
        candidates.append(image_dir / f"{raw_path.stem}.npy")
    elif raw_path.suffix.lower() == ".npy":
        candidates.append(image_dir / f"{raw_path.stem}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Image file not found. Searched: {searched}")


def load_image_array(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        image = np.load(path)
    else:
        with Image.open(path) as img:
            image = np.array(img.convert("RGB"))

    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.moveaxis(image, 0, -1)

    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape {image.shape} for {path}")

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[-1] > 3:
        image = image[..., :3]

    image = image.astype(np.float32)
    if image.size and np.nanmax(image) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0.0, 255.0).astype(np.uint8)


def preload_images_to_ram(dataframe, image_path_column: str):
    cache = []
    for idx, path in enumerate(dataframe[image_path_column].tolist(), start=1):
        cache.append(load_image_array(Path(path)))
        if idx % 5000 == 0:
            print(f"Loaded {idx:,} images into RAM")
    return cache


def load_or_build_image_cache(name: str, dataframe, image_path_column: str, cache_dir: Path, use_disk_cache: bool):
    cache_path = cache_dir / f"{name}_images_cache.npy"
    if use_disk_cache and cache_path.exists():
        print(f"Loading {name} image cache from {cache_path}")
        return np.load(cache_path)

    print(f"Building {name} image cache")
    cache = preload_images_to_ram(dataframe, image_path_column)

    if use_disk_cache:
        try:
            stacked = np.stack(cache, axis=0)
        except ValueError:
            print(f"Skipping disk cache for {name}; images do not share one shape.")
            return cache

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, stacked)
        print(f"Saved {name} image cache to {cache_path}")
        return stacked

    return cache


def make_resize_transform(resize_size: int):
    try:
        resize = transforms.Resize((resize_size, resize_size), antialias=True)
    except TypeError:
        resize = transforms.Resize((resize_size, resize_size))

    return transforms.Compose(
        [
            resize,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def make_dataset_class():
    class CachedLayoutRiskDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, image_cache, target_column: str, transform):
            self.dataframe = dataframe.reset_index(drop=True)
            self.images = image_cache
            self.targets = self.dataframe[target_column].astype(float).to_numpy(dtype=np.float32)
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            array = self.images[idx]
            tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 255.0
            if self.transform is not None:
                tensor = self.transform(tensor)
            target = torch.tensor(self.targets[idx], dtype=torch.float32)
            return tensor, target

    return CachedLayoutRiskDataset


def split_by_layout(dataframe, layout_column: str, split_seed: int):
    layout_ids = dataframe[layout_column].drop_duplicates().to_numpy()
    if len(layout_ids) < 3:
        raise ValueError("At least three unique layouts are required for train/val/test splits.")

    shuffled = layout_ids.copy()
    rng = np.random.RandomState(split_seed)
    rng.shuffle(shuffled)

    n_layouts = len(shuffled)
    n_train = int(0.8 * n_layouts)
    n_val = int(0.1 * n_layouts)

    if n_train < 1:
        n_train = 1
    if n_val < 1:
        n_val = 1
    if n_train + n_val >= n_layouts:
        n_train = max(1, n_layouts - 2)
        n_val = 1

    train_layouts = shuffled[:n_train]
    val_layouts = shuffled[n_train : n_train + n_val]
    test_layouts = shuffled[n_train + n_val :]

    train_df = dataframe[dataframe[layout_column].isin(train_layouts)].reset_index(drop=True)
    val_df = dataframe[dataframe[layout_column].isin(val_layouts)].reset_index(drop=True)
    test_df = dataframe[dataframe[layout_column].isin(test_layouts)].reset_index(drop=True)

    return train_df, val_df, test_df, train_layouts, val_layouts, test_layouts


def save_split_metadata(output_dir: Path, train_layouts, val_layouts, test_layouts) -> None:
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"layout_id": train_layouts}).to_csv(split_dir / "train_layout_ids.csv", index=False)
    pd.DataFrame({"layout_id": val_layouts}).to_csv(split_dir / "val_layout_ids.csv", index=False)
    pd.DataFrame({"layout_id": test_layouts}).to_csv(split_dir / "test_layout_ids.csv", index=False)


def prepare_datasets(args, paths):
    csv_path, image_dir, output_dir = paths
    dataframe = pd.read_csv(csv_path)

    required_columns = [
        args.layout_column,
        args.image_column,
        args.target_column,
        args.clean_column,
    ]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    clean_df = dataframe[clean_success_mask(dataframe[args.clean_column])].copy()
    clean_df = clean_df.reset_index(drop=True)
    if clean_df.empty:
        raise ValueError("No clean-success rows found.")

    clean_df["resolved_image_path"] = clean_df[args.image_column].apply(
        lambda value: str(resolve_image_path(value, image_dir))
    )

    train_df, val_df, test_df, train_layouts, val_layouts, test_layouts = split_by_layout(
        clean_df, args.layout_column, args.split_seed
    )
    save_split_metadata(output_dir, train_layouts, val_layouts, test_layouts)

    target_mean = float(train_df[args.target_column].astype(float).mean())
    target_std = float(train_df[args.target_column].astype(float).std())
    if not math.isfinite(target_std) or target_std <= 0:
        raise ValueError("Training target standard deviation must be positive.")

    save_json(
        output_dir / "target_stats.json",
        {
            "target_column": args.target_column,
            "target_mean": target_mean,
            "target_std": target_std,
            "clean_rows": int(len(clean_df)),
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "train_layouts": int(len(train_layouts)),
            "val_layouts": int(len(val_layouts)),
            "test_layouts": int(len(test_layouts)),
            "split_seed": int(args.split_seed),
        },
    )

    print(f"Clean-success rows: {len(clean_df):,}")
    print(f"Train/val/test rows: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}")
    print(f"Train target mean/std: {target_mean:.6f} / {target_std:.6f}")

    cache_dir = output_dir / "cache"
    train_cache = load_or_build_image_cache(
        "train",
        train_df,
        "resolved_image_path",
        cache_dir,
        args.cache_images,
    )
    val_cache = load_or_build_image_cache(
        "val",
        val_df,
        "resolved_image_path",
        cache_dir,
        args.cache_images,
    )
    test_cache = load_or_build_image_cache(
        "test",
        test_df,
        "resolved_image_path",
        cache_dir,
        args.cache_images,
    )

    transform = make_resize_transform(args.resize_size)
    dataset_cls = make_dataset_class()
    train_dataset = dataset_cls(train_df, train_cache, args.target_column, transform)
    val_dataset = dataset_cls(val_df, val_cache, args.target_column, transform)
    test_dataset = dataset_cls(test_df, test_cache, args.target_column, transform)

    return train_dataset, val_dataset, test_dataset, test_df, target_mean, target_std


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int, device):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )


def build_resnet18(hidden_dim: int, dropout: float, pretrained: bool):
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except AttributeError:
            model = models.resnet18(pretrained=True)
    else:
        try:
            model = models.resnet18(weights=None)
        except TypeError:
            model = models.resnet18(pretrained=False)

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    return model


def build_vit_tiny(hidden_dim: int, dropout: float, pretrained: bool):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=pretrained, num_classes=0)
    in_features = model.num_features
    model.head = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    return model


def build_model(model_name: str, hidden_dim: int, dropout: float, pretrained: bool):
    if model_name == "resnet18":
        return build_resnet18(hidden_dim, dropout, pretrained)
    if model_name == "vit_tiny":
        return build_vit_tiny(hidden_dim, dropout, pretrained)
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_target(values, target_mean: float, target_std: float):
    return (values - target_mean) / target_std


def denormalize_target(values, target_mean: float, target_std: float):
    return values * target_std + target_mean


def r2_score_np(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def train_one_epoch(model, loader, criterion, optimizer, device, target_mean: float, target_std: float):
    model.train()
    total_loss = 0.0
    total_count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        normalized_targets = normalize_target(targets, target_mean, target_std)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images).squeeze(-1)
        loss = criterion(outputs, normalized_targets)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def evaluate(model, loader, criterion, device, target_mean: float, target_std: float):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            normalized_targets = normalize_target(targets, target_mean, target_std)

            outputs = model(images).squeeze(-1)
            loss = criterion(outputs, normalized_targets)

            batch_size = images.size(0)
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            predictions = denormalize_target(outputs.detach().cpu().numpy(), target_mean, target_std)
            all_preds.append(predictions)
            all_targets.append(targets.detach().cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    residual_bias = float(np.mean(y_pred - y_true))
    r2 = r2_score_np(y_true, y_pred)

    return {
        "loss": total_loss / max(total_count, 1),
        "mae": mae,
        "rmse": rmse,
        "residual_bias": residual_bias,
        "r2": r2,
        "predictions": y_pred,
        "targets": y_true,
    }


def move_optimizer_state_to_device(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_checkpoint(path: Path, model, optimizer, scheduler, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    move_optimizer_state_to_device(optimizer, device)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    scheduler,
    epoch: int,
    history: list[dict],
    best_val_loss: float,
    best_epoch: int,
    best_state_dict,
    no_improve_count: int,
    model_name: str,
    train_seed: int,
    split_seed: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": int(epoch),
            "history": history,
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(best_epoch),
            "best_state_dict": best_state_dict,
            "no_improve_count": int(no_improve_count),
            "model_name": model_name,
            "train_seed": int(train_seed),
            "split_seed": int(split_seed),
        },
        path,
    )


def train_full(
    model_name: str,
    model,
    learning_rate: float,
    weight_decay: float,
    train_dataset,
    val_dataset,
    args,
    device,
    target_mean: float,
    target_std: float,
    train_seed: int,
    save_path: Path | None,
    checkpoint_path: Path | None,
):
    train_loader = make_dataloader(
        train_dataset,
        args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=train_seed,
        device=device,
    )
    val_loader = make_dataloader(
        val_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.split_seed,
        device=device,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    history: list[dict] = []
    no_improve_count = 0
    start_epoch = 0

    if args.resume and checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler, device)
        start_epoch = int(checkpoint["epoch"])
        history = checkpoint["history"]
        best_val_loss = float(checkpoint["best_val_loss"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state_dict = checkpoint["best_state_dict"]
        no_improve_count = int(checkpoint["no_improve_count"])
        print(f"[{model_name}] Resuming from epoch {start_epoch}.")

    print(
        f"[{model_name}] seed={train_seed} lr={learning_rate:g} wd={weight_decay:g} "
        f"max_epochs={args.max_epochs}"
    )

    start_time = time.time()
    for epoch in range(start_epoch, args.max_epochs):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            target_mean,
            target_std,
        )
        val_metrics = evaluate(model, val_loader, criterion, device, target_mean, target_std)
        scheduler.step(val_metrics["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_number = epoch + 1

        record = {
            "epoch": int(epoch_number),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_mae": float(val_metrics["mae"]),
            "val_rmse": float(val_metrics["rmse"]),
            "val_r2": float(val_metrics["r2"]),
            "learning_rate": current_lr,
            "epoch_seconds": float(time.time() - epoch_start),
        }
        history.append(record)

        is_best = val_metrics["loss"] < best_val_loss - args.min_delta
        if is_best:
            best_val_loss = float(val_metrics["loss"])
            best_epoch = int(epoch_number)
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            no_improve_count = 0
        else:
            no_improve_count += 1

        marker = " best" if is_best else ""
        print(
            f"[{model_name}] epoch {epoch_number:03d}/{args.max_epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_mae={val_metrics['mae']:.4f} val_r2={val_metrics['r2']:.4f} "
            f"lr={current_lr:.1e}{marker}"
        )

        if (
            checkpoint_path is not None
            and args.checkpoint_every > 0
            and epoch_number % args.checkpoint_every == 0
        ):
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch_number,
                history,
                best_val_loss,
                best_epoch,
                best_state_dict,
                no_improve_count,
                model_name,
                train_seed,
                args.split_seed,
            )
            print(f"[{model_name}] Saved resume checkpoint at epoch {epoch_number}.")

        if no_improve_count >= args.early_stop_patience:
            print(f"[{model_name}] Early stopping at epoch {epoch_number}.")
            break

    total_seconds = time.time() - start_time
    if best_state_dict is None:
        raise RuntimeError(f"No best state was recorded for {model_name}.")

    model.load_state_dict(best_state_dict)

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state_dict,
                "model_name": model_name,
                "best_epoch": int(best_epoch),
                "total_epochs_run": int(len(history)),
                "best_val_loss": float(best_val_loss),
                "hyperparameters": {
                    "learning_rate": float(learning_rate),
                    "weight_decay": float(weight_decay),
                    "max_epochs": int(args.max_epochs),
                    "batch_size": int(args.batch_size),
                    "early_stop_patience": int(args.early_stop_patience),
                    "scheduler_patience": int(args.scheduler_patience),
                    "hidden_dim": int(args.hidden_dim),
                    "dropout": float(args.dropout),
                    "pretrained": bool(args.pretrained),
                },
                "target_mean": float(target_mean),
                "target_std": float(target_std),
                "history": history,
                "split_seed": int(args.split_seed),
                "train_seed": int(train_seed),
            },
            save_path,
        )
        print(f"[{model_name}] Saved best model to {save_path}")

    if checkpoint_path is not None and checkpoint_path.exists() and not args.keep_checkpoints:
        checkpoint_path.unlink()
        print(f"[{model_name}] Removed completed resume checkpoint {checkpoint_path}")

    return {
        "model": model_name,
        "best_epoch": int(best_epoch),
        "total_epochs_run": int(len(history)),
        "best_val_loss": float(best_val_loss),
        "total_training_seconds": float(total_seconds),
        "history": history,
    }


def save_predictions(path: Path, test_df, args, train_seed: int, model_name: str, metrics) -> None:
    metadata_columns = [
        column
        for column in [args.layout_column, "trial_id", "category", "resolved_image_path"]
        if column in test_df.columns
    ]
    output = test_df[metadata_columns].copy()
    output["train_seed"] = train_seed
    output["model"] = model_name
    output["target"] = metrics["targets"]
    output["prediction"] = metrics["predictions"]
    output["residual"] = metrics["predictions"] - metrics["targets"]
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def load_model_config(path: Path):
    config = load_json(path)
    learning_rate = config.get("learning_rate", config.get("lr"))
    weight_decay = config.get("weight_decay", config.get("wd"))
    if learning_rate is None or weight_decay is None:
        raise ValueError(f"Config must include learning_rate and weight_decay: {path}")
    return {"learning_rate": float(learning_rate), "weight_decay": float(weight_decay)}


def find_search_config(config_dir: Path, model_name: str) -> Path | None:
    candidates = [
        config_dir / model_name / f"{model_name}_best_config.json",
        config_dir / f"{model_name}_best_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_hyperparameters(args, model_names: list[str]):
    configs = {name: dict(DEFAULT_HYPERPARAMETERS[name]) for name in model_names}

    if args.config_dir is not None:
        for model_name in model_names:
            config_path = find_search_config(args.config_dir, model_name)
            if config_path is not None:
                configs[model_name].update(load_model_config(config_path))

    explicit_paths = {
        "resnet18": args.resnet18_config,
        "vit_tiny": args.vit_tiny_config,
    }
    for model_name, config_path in explicit_paths.items():
        if model_name in configs and config_path is not None:
            configs[model_name].update(load_model_config(config_path))

    explicit_values = {
        "resnet18": {
            "learning_rate": args.resnet18_lr,
            "weight_decay": args.resnet18_weight_decay,
        },
        "vit_tiny": {
            "learning_rate": args.vit_tiny_lr,
            "weight_decay": args.vit_tiny_weight_decay,
        },
    }
    for model_name in model_names:
        for key, value in explicit_values[model_name].items():
            if value is not None:
                configs[model_name][key] = float(value)

    return configs


def test_results_path(seed_dir: Path) -> Path:
    return seed_dir / "test_results.json"


def load_seed_results(seed_dir: Path, train_seed: int, split_seed: int):
    path = test_results_path(seed_dir)
    if path.exists():
        results = load_json(path)
        if "models" in results:
            return results

    return {
        "train_seed": int(train_seed),
        "split_seed": int(split_seed),
        "models": {},
    }


def train_and_evaluate_seed(
    train_seed: int,
    model_names: list[str],
    model_configs: dict,
    train_dataset,
    val_dataset,
    test_dataset,
    test_df,
    args,
    device,
    target_mean: float,
    target_std: float,
):
    seed_dir = args.output_dir / f"seed_{train_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_results = load_seed_results(seed_dir, train_seed, args.split_seed)

    test_loader = make_dataloader(
        test_dataset,
        args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        seed=args.split_seed,
        device=device,
    )
    criterion = nn.MSELoss()

    for model_name in model_names:
        if (
            not args.overwrite
            and model_name in seed_results.get("models", {})
            and test_results_path(seed_dir).exists()
        ):
            print(f"[{model_name}] seed={train_seed} already has test results; skipping.")
            continue

        set_seed(train_seed, deterministic=args.deterministic)
        model = build_model(model_name, args.hidden_dim, args.dropout, args.pretrained).to(device)

        config = model_configs[model_name]
        save_path = None
        if args.save_models:
            save_path = seed_dir / f"{model_name}_layout_risk.pt"
        checkpoint_path = seed_dir / f"{model_name}_checkpoint_latest.pt"

        training_info = train_full(
            model_name,
            model,
            config["learning_rate"],
            config["weight_decay"],
            train_dataset,
            val_dataset,
            args,
            device,
            target_mean,
            target_std,
            train_seed,
            save_path,
            checkpoint_path,
        )

        test_metrics = evaluate(model, test_loader, criterion, device, target_mean, target_std)
        if args.save_predictions:
            save_predictions(
                seed_dir / f"{model_name}_test_predictions.csv",
                test_df,
                args,
                train_seed,
                model_name,
                test_metrics,
            )

        seed_results["models"][model_name] = {
            "mae": float(test_metrics["mae"]),
            "rmse": float(test_metrics["rmse"]),
            "residual_bias": float(test_metrics["residual_bias"]),
            "r2": float(test_metrics["r2"]),
            "best_epoch": int(training_info["best_epoch"]),
            "total_epochs_run": int(training_info["total_epochs_run"]),
            "best_val_loss": float(training_info["best_val_loss"]),
            "learning_rate": float(config["learning_rate"]),
            "weight_decay": float(config["weight_decay"]),
            "model_path": str(save_path) if save_path is not None else None,
        }
        save_json(test_results_path(seed_dir), seed_results)
        save_json(seed_dir / f"{model_name}_training_history.json", training_info["history"])

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return seed_results


def collect_seed_results(output_dir: Path):
    rows = []
    pattern = str(output_dir / "seed_*" / "test_results.json")
    for path_text in sorted(glob.glob(pattern)):
        result = load_json(Path(path_text))
        for model_name, metrics in result.get("models", {}).items():
            row = {
                "train_seed": int(result["train_seed"]),
                "split_seed": int(result["split_seed"]),
                "model": model_name,
            }
            row.update(metrics)
            rows.append(row)

    if not rows:
        return None, None

    raw_df = pd.DataFrame(rows)
    metric_columns = ["mae", "rmse", "residual_bias", "r2", "best_epoch", "total_epochs_run"]
    available_metrics = [column for column in metric_columns if column in raw_df.columns]
    stats_df = raw_df.groupby("model")[available_metrics].agg(["mean", "std"])
    return raw_df, stats_df


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run final multi-seed training for ResNet-18 and ViT-Tiny."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated_dataset"))
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multi_seed_training"))
    parser.add_argument("--models", choices=["resnet18", "vit_tiny", "all"], default="all")

    parser.add_argument("--layout-column", default="layout_id")
    parser.add_argument("--image-column", default="layout_image_path")
    parser.add_argument("--target-column", default="invaded_grid_count")
    parser.add_argument("--clean-column", default="clean_success")

    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-seeds", type=parse_int_list, default=parse_int_list("42,43,44"))
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--resize-size", type=int, default=224)

    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default=None)

    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--resnet18-config", type=Path, default=None)
    parser.add_argument("--vit-tiny-config", type=Path, default=None)
    parser.add_argument("--resnet18-lr", type=float, default=None)
    parser.add_argument("--resnet18-weight-decay", type=float, default=None)
    parser.add_argument("--vit-tiny-lr", type=float, default=None)
    parser.add_argument("--vit-tiny-weight-decay", type=float, default=None)

    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-cache-images", action="store_false", dest="cache_images")
    parser.add_argument("--no-save-models", action="store_false", dest="save_models")
    parser.add_argument("--no-save-predictions", action="store_false", dest="save_predictions")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.set_defaults(
        pretrained=True,
        resume=True,
        cache_images=True,
        save_models=True,
        save_predictions=True,
    )
    return parser.parse_args(argv)


def resolve_paths(args):
    csv_path = args.csv_path if args.csv_path is not None else args.data_dir / "trial_summary_all.csv"
    image_dir = args.image_dir if args.image_dir is not None else args.data_dir / "layout_images"
    return csv_path, image_dir, args.output_dir


def main(argv=None) -> int:
    args = parse_args(argv)
    model_names = ["resnet18", "vit_tiny"] if args.models == "all" else [args.models]

    load_libraries(require_timm="vit_tiny" in model_names)
    set_seed(args.split_seed, deterministic=args.deterministic)

    csv_path, image_dir, output_dir = resolve_paths(args)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    args.output_dir = output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    print(f"CSV: {csv_path}")
    print(f"Images: {image_dir}")
    print(f"Output: {output_dir}")

    model_configs = resolve_hyperparameters(args, model_names)
    save_json(output_dir / "training_config.json", {
        "model_configs": model_configs,
        "split_seed": int(args.split_seed),
        "train_seeds": [int(seed) for seed in args.train_seeds],
        "max_epochs": int(args.max_epochs),
        "batch_size": int(args.batch_size),
        "early_stop_patience": int(args.early_stop_patience),
        "scheduler_patience": int(args.scheduler_patience),
        "checkpoint_every": int(args.checkpoint_every),
        "pretrained": bool(args.pretrained),
    })

    train_dataset, val_dataset, test_dataset, test_df, target_mean, target_std = prepare_datasets(
        args,
        (csv_path, image_dir, output_dir),
    )

    for train_seed in args.train_seeds:
        print(f"Running training seed {train_seed} with fixed split seed {args.split_seed}")
        train_and_evaluate_seed(
            train_seed,
            model_names,
            model_configs,
            train_dataset,
            val_dataset,
            test_dataset,
            test_df,
            args,
            device,
            target_mean,
            target_std,
        )

    raw_df, stats_df = collect_seed_results(output_dir)
    if raw_df is not None:
        raw_path = output_dir / "seed_summary_raw.csv"
        stats_path = output_dir / "seed_summary_stats.csv"
        raw_df.to_csv(raw_path, index=False)
        stats_df.to_csv(stats_path)
        print(f"Saved seed summaries to {raw_path} and {stats_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
