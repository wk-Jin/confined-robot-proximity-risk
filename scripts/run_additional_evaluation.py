"""
Run additional evaluation for multi-seed trained models.

This script evaluates ResNet-18 and ViT-Tiny checkpoints on the additional
C0-C4 evaluation dataset. It saves predictions, overall metrics, category-level
metrics, risk-level metrics, and mean/std summaries
across training seeds.

Visualization code is intentionally kept out of this file. The saved CSV files
are intended to be reused by a separate paper-figure script.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
from pathlib import Path


np = None
pd = None
Image = None
torch = None
nn = None
transforms = None
models = None
timm = None
stats = None
wilcoxon = None


MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "vit_tiny": "ViT-Tiny",
}

MODEL_FILE_NAMES = {
    "resnet18": ["resnet18_layout_risk.pt"],
    "vit_tiny": ["vit_tiny_layout_risk.pt", "vit_layout_risk.pt"],
}

DEFAULT_CATEGORY_ORDER = [
    "C0",
    "C1",
    "C2",
    "C3",
    "C4",
    "C0_interp",
    "C1_1axis",
    "C2_2axis",
    "C3_3axis",
    "C4_all",
]

RISK_LEVELS = ["Low", "Medium", "High", "Extreme"]


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one integer seed.")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one category.")
    return values


def load_libraries(require_timm: bool = False, require_wilcoxon: bool = False) -> None:
    """Import heavy libraries after CLI parsing so --help stays lightweight."""
    global np, pd, Image, torch, nn, transforms, models, timm, stats, wilcoxon

    try:
        import numpy as np_module
        import pandas as pd_module
        from PIL import Image as image_module
        import torch as torch_module
        from torch import nn as nn_module
        from torchvision import models as models_module
        from torchvision import transforms as transforms_module
        from scipy import stats as stats_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install the project requirements "
            "before running additional evaluation."
        ) from exc

    np = np_module
    pd = pd_module
    Image = image_module
    torch = torch_module
    nn = nn_module
    transforms = transforms_module
    models = models_module
    stats = stats_module
    wilcoxon = None

    if require_wilcoxon:
        try:
            from scipy.stats import wilcoxon as wilcoxon_function
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: scipy. Install scipy to run Wilcoxon tests.") from exc
        wilcoxon = wilcoxon_function

    if require_timm:
        try:
            import timm as timm_module
        except ModuleNotFoundError as exc:
            raise SystemExit("Missing dependency: timm. Install timm to evaluate ViT-Tiny.") from exc
        timm = timm_module


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def find_default_csv(data_dir: Path) -> Path:
    candidates = [
        data_dir / "additional_evaluation_dataset.csv",
        data_dir / "phase2_trial_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def category_sort_key(category: str) -> tuple[int, str]:
    text = str(category)
    if text.startswith("C") and len(text) >= 2 and text[1].isdigit():
        return int(text[1]), text
    return 99, text


def infer_category_order(dataframe, category_column: str, requested_order: list[str] | None):
    observed = list(dataframe[category_column].dropna().astype(str).unique())
    if requested_order is not None:
        requested = [cat for cat in requested_order if cat in observed]
        remainder = [cat for cat in observed if cat not in requested]
        return requested + sorted(remainder, key=category_sort_key)

    preferred = [cat for cat in DEFAULT_CATEGORY_ORDER if cat in observed]
    remainder = [cat for cat in observed if cat not in preferred]
    return preferred + sorted(remainder, key=category_sort_key)


def validate_eval_dataset(dataframe, category_order: list[str], args) -> None:
    required_columns = [args.category_column, args.image_column, args.target_column]
    missing_columns = [column for column in required_columns if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Evaluation CSV is missing required columns: {missing_columns}")

    if not args.strict_counts:
        return

    if args.expected_total > 0 and len(dataframe) != args.expected_total:
        raise ValueError(
            f"Expected {args.expected_total} evaluation rows, found {len(dataframe)}."
        )

    if args.expected_per_category > 0:
        counts = dataframe[args.category_column].value_counts()
        for category in category_order:
            n_rows = int(counts.get(category, 0))
            if n_rows != args.expected_per_category:
                raise ValueError(
                    f"Expected {args.expected_per_category} rows for {category}, found {n_rows}."
                )


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


def load_eval_images(dataframe, image_path_column: str):
    images = []
    for idx, path in enumerate(dataframe[image_path_column].tolist(), start=1):
        images.append(load_image_array(Path(path)))
        if idx % 1000 == 0:
            print(f"Loaded {idx:,} evaluation images")
    return images


def load_or_build_image_cache(dataframe, image_path_column: str, cache_path: Path | None):
    if cache_path is not None and cache_path.exists():
        print(f"Loading evaluation image cache from {cache_path}")
        return list(np.load(cache_path))

    print("Building evaluation image cache")
    images = load_eval_images(dataframe, image_path_column)

    if cache_path is not None:
        try:
            stacked = np.stack(images, axis=0)
        except ValueError:
            print("Skipping disk cache because evaluation images do not share one shape.")
            return images

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, stacked)
        print(f"Saved evaluation image cache to {cache_path}")
        return list(stacked)

    return images


def make_preprocess_transform(resize_size: int):
    try:
        resize = transforms.Resize((resize_size, resize_size), antialias=True)
    except TypeError:
        resize = transforms.Resize((resize_size, resize_size))

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return resize, normalize


def preprocess_batch(images, resize_transform, normalize_transform):
    batch = np.stack(images, axis=0)
    tensor = torch.from_numpy(batch).float() / 255.0
    tensor = tensor.permute(0, 3, 1, 2)
    tensor = resize_transform(tensor)
    tensor = normalize_transform(tensor)
    return tensor


def build_resnet18(hidden_dim: int, dropout: float):
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


def build_vit_tiny(hidden_dim: int, dropout: float):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    in_features = model.num_features
    model.head = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    return model


def build_model(model_name: str, hidden_dim: int, dropout: float):
    if model_name == "resnet18":
        return build_resnet18(hidden_dim, dropout)
    if model_name == "vit_tiny":
        return build_vit_tiny(hidden_dim, dropout)
    raise ValueError(f"Unsupported model: {model_name}")


def adapt_state_dict(state_dict):
    if not any(key.startswith("backbone.") for key in state_dict):
        return state_dict

    adapted = {}
    for key, value in state_dict.items():
        if key.startswith("backbone."):
            adapted[key[len("backbone.") :]] = value
        else:
            adapted[key] = value
    return adapted


def torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def find_seed_model_dir(model_base_dir: Path, train_seed: int) -> Path:
    candidates = [
        model_base_dir / f"seed_{train_seed}",
        model_base_dir / f"seed{train_seed}",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_model_path(seed_model_dir: Path, model_name: str) -> Path:
    for file_name in MODEL_FILE_NAMES[model_name]:
        candidate = seed_model_dir / file_name
        if candidate.exists():
            return candidate
    expected = ", ".join(MODEL_FILE_NAMES[model_name])
    raise FileNotFoundError(f"Could not find {model_name} checkpoint in {seed_model_dir}. Expected: {expected}")


def load_trained_model(model_name: str, model_path: Path, args, device):
    checkpoint = torch_load(model_path, device)
    if "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {model_path}")

    if "target_mean" not in checkpoint or "target_std" not in checkpoint:
        raise ValueError(f"Checkpoint must contain target_mean and target_std: {model_path}")

    model = build_model(model_name, args.hidden_dim, args.dropout).to(device)
    state_dict = adapt_state_dict(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    return model, float(checkpoint["target_mean"]), float(checkpoint["target_std"])


def predict_model(model, images, batch_size: int, device, target_mean: float, target_std: float, transforms_pair):
    resize_transform, normalize_transform = transforms_pair
    predictions = []

    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            batch = preprocess_batch(batch_images, resize_transform, normalize_transform).to(device)
            normalized = model(batch).squeeze(-1).detach().cpu().numpy()
            predictions.append(normalized * target_std + target_mean)

    return np.concatenate(predictions)


def assign_risk_levels(true_values):
    series = pd.Series(true_values)
    q25 = float(series.quantile(0.25))
    q75 = float(series.quantile(0.75))
    q90 = float(series.quantile(0.90))

    labels = []
    for value in series:
        if value <= q25:
            labels.append("Low")
        elif value <= q75:
            labels.append("Medium")
        elif value <= q90:
            labels.append("High")
        else:
            labels.append("Extreme")

    return labels, {"q25": q25, "q75": q75, "q90": q90}


def r2_score_np(true_values, predictions) -> float:
    true_values = np.asarray(true_values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    ss_res = np.sum((true_values - predictions) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def upr_percent(true_values, predictions) -> float:
    true_values = np.asarray(true_values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    return float(np.mean(true_values > predictions) * 100.0)


def safe_linregress(true_values, predictions):
    true_values = np.asarray(true_values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(true_values) < 2 or np.allclose(true_values, true_values[0]):
        return float("nan"), float("nan")
    slope, intercept, _, _, _ = stats.linregress(true_values, predictions)
    return float(slope), float(intercept)


def safe_spearman(true_values, predictions):
    true_values = np.asarray(true_values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if len(true_values) < 2:
        return float("nan")
    if np.allclose(true_values, true_values[0]) or np.allclose(predictions, predictions[0]):
        return float("nan")
    coefficient, _ = stats.spearmanr(true_values, predictions)
    return float(coefficient)


def compute_regression_metrics(true_values, predictions):
    true_values = np.asarray(true_values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - true_values
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "me": float(np.mean(errors)),
        "r2": r2_score_np(true_values, predictions),
        "upr": upr_percent(true_values, predictions),
    }


def build_prediction_dataframe(eval_df, args, category_order: list[str], predictions_by_model: dict):
    metadata_columns = [
        column
        for column in [
            args.category_column,
            "candidate_id",
            "combination_id",
            "n_extrapolated_groups",
            "fence_x",
            "pillar_x",
            "pillar_y",
            "pick_x_offset",
            "pick_y_offset",
            "place_x_offset",
            "place_y_offset",
            "fence_extrap",
            "pillar_extrap",
            "pick_extrap",
            "place_extrap",
            "resolved_image_path",
            args.image_column,
        ]
        if column in eval_df.columns
    ]
    pred_df = eval_df[metadata_columns].copy()
    pred_df["true_value"] = eval_df[args.target_column].astype(float).to_numpy()

    risk_levels, thresholds = assign_risk_levels(pred_df["true_value"])
    pred_df["risk_level"] = pd.Categorical(risk_levels, categories=RISK_LEVELS, ordered=True)

    for model_name, predictions in predictions_by_model.items():
        pred_df[f"{model_name}_pred"] = predictions
        pred_df[f"{model_name}_error"] = predictions - pred_df["true_value"]
        pred_df[f"{model_name}_abs_error"] = np.abs(pred_df[f"{model_name}_error"])

    category_dtype = pd.CategoricalDtype(categories=category_order, ordered=True)
    pred_df[args.category_column] = pred_df[args.category_column].astype(str).astype(category_dtype)
    pred_df = pred_df.sort_values([args.category_column]).reset_index(drop=True)
    return pred_df, thresholds


def compute_overall_metrics(pred_df, model_names: list[str]):
    rows = []
    true_values = pred_df["true_value"].to_numpy()
    for model_name in model_names:
        predictions = pred_df[f"{model_name}_pred"].to_numpy()
        row = {
            "model": model_name,
            "model_label": MODEL_LABELS[model_name],
            "n": int(len(pred_df)),
        }
        row.update(compute_regression_metrics(true_values, predictions))
        rows.append(row)
    return pd.DataFrame(rows)


def compute_category_metrics(pred_df, model_names: list[str], category_column: str, category_order: list[str]):
    rows = []
    for category in category_order:
        sub = pred_df[pred_df[category_column] == category]
        if sub.empty:
            continue

        true_values = sub["true_value"].to_numpy()
        for model_name in model_names:
            predictions = sub[f"{model_name}_pred"].to_numpy()
            slope, intercept = safe_linregress(true_values, predictions)
            row = {
                "category": category,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "n": int(len(sub)),
                "slope": slope,
                "intercept": intercept,
                "spearman": safe_spearman(true_values, predictions),
            }
            row.update(compute_regression_metrics(true_values, predictions))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_risk_level_metrics(pred_df, model_names: list[str]):
    rows = []
    for risk_level in RISK_LEVELS:
        sub = pred_df[pred_df["risk_level"] == risk_level]
        if sub.empty:
            continue

        true_values = sub["true_value"].to_numpy()
        for model_name in model_names:
            predictions = sub[f"{model_name}_pred"].to_numpy()
            row = {
                "risk_level": risk_level,
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
                "n": int(len(sub)),
            }
            row.update(compute_regression_metrics(true_values, predictions))
            rows.append(row)
    return pd.DataFrame(rows)


def safe_wilcoxon(abs_error_a, abs_error_b):
    if wilcoxon is None:
        raise RuntimeError("Wilcoxon tests require --run-wilcoxon.")
    a = np.asarray(abs_error_a, dtype=np.float64)
    b = np.asarray(abs_error_b, dtype=np.float64)
    n = int(len(a))
    if n == 0:
        return {"n": 0, "statistic": float("nan"), "p_value": float("nan")}

    diff = a - b
    if np.allclose(diff, 0.0):
        return {"n": n, "statistic": 0.0, "p_value": 1.0}

    try:
        statistic, p_value = wilcoxon(a, b)
    except ValueError:
        return {"n": n, "statistic": float("nan"), "p_value": float("nan")}

    return {"n": n, "statistic": float(statistic), "p_value": float(p_value)}


def compute_wilcoxon_tests(pred_df, category_column: str, category_order: list[str]):
    if "resnet18_abs_error" not in pred_df.columns or "vit_tiny_abs_error" not in pred_df.columns:
        return pd.DataFrame()

    rows = []

    overall = safe_wilcoxon(pred_df["resnet18_abs_error"], pred_df["vit_tiny_abs_error"])
    rows.append({"scope": "overall", "group": "overall", **overall})

    for category in category_order:
        sub = pred_df[pred_df[category_column] == category]
        result = safe_wilcoxon(sub["resnet18_abs_error"], sub["vit_tiny_abs_error"])
        rows.append({"scope": "category", "group": category, **result})

    for risk_level in RISK_LEVELS:
        sub = pred_df[pred_df["risk_level"] == risk_level]
        result = safe_wilcoxon(sub["resnet18_abs_error"], sub["vit_tiny_abs_error"])
        rows.append({"scope": "risk_level", "group": risk_level, **result})

    return pd.DataFrame(rows)


def evaluate_seed(
    train_seed: int,
    model_names: list[str],
    eval_df,
    images,
    category_order: list[str],
    args,
    device,
    transforms_pair,
):
    seed_output_dir = args.output_dir / f"seed_{train_seed}"
    summary_path = seed_output_dir / "metrics_summary.json"
    if summary_path.exists() and not args.overwrite:
        print(f"Skipping completed seed {train_seed}: {seed_output_dir}")
        return

    seed_model_dir = find_seed_model_dir(args.model_dir, train_seed)
    if not seed_model_dir.exists():
        raise FileNotFoundError(f"Model directory not found for seed {train_seed}: {seed_model_dir}")

    seed_output_dir.mkdir(parents=True, exist_ok=True)
    predictions_by_model = {}
    model_metadata = {}

    for model_name in model_names:
        model_path = find_model_path(seed_model_dir, model_name)
        model, target_mean, target_std = load_trained_model(model_name, model_path, args, device)
        predictions_by_model[model_name] = predict_model(
            model,
            images,
            args.batch_size,
            device,
            target_mean,
            target_std,
            transforms_pair,
        )
        model_metadata[model_name] = {
            "model_path": str(model_path),
            "target_mean": target_mean,
            "target_std": target_std,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pred_df, thresholds = build_prediction_dataframe(
        eval_df,
        args,
        category_order,
        predictions_by_model,
    )
    overall_df = compute_overall_metrics(pred_df, model_names)
    category_df = compute_category_metrics(pred_df, model_names, args.category_column, category_order)
    risk_level_df = compute_risk_level_metrics(pred_df, model_names)
    wilcoxon_df = (
        compute_wilcoxon_tests(pred_df, args.category_column, category_order)
        if args.run_wilcoxon and set(model_names) == {"resnet18", "vit_tiny"}
        else pd.DataFrame()
    )

    pred_df.to_csv(seed_output_dir / "all_predictions.csv", index=False)
    overall_df.to_csv(seed_output_dir / "overall_metrics.csv", index=False)
    category_df.to_csv(seed_output_dir / "category_metrics.csv", index=False)
    risk_level_df.to_csv(seed_output_dir / "risk_level_metrics.csv", index=False)
    if not wilcoxon_df.empty:
        wilcoxon_df.to_csv(seed_output_dir / "wilcoxon_signed_rank.csv", index=False)

    summary = {
        "train_seed": int(train_seed),
        "models": model_metadata,
        "risk_thresholds": thresholds,
        "overall_metrics": overall_df.to_dict("records"),
    }
    if not wilcoxon_df.empty:
        summary["wilcoxon_signed_rank"] = wilcoxon_df.to_dict("records")
    save_json(summary_path, summary)
    print(f"Saved additional evaluation results for seed {train_seed}: {seed_output_dir}")


def add_train_seed_column(dataframe, seed: int):
    dataframe = dataframe.copy()
    dataframe.insert(0, "train_seed", int(seed))
    return dataframe


def aggregate_metric_file(output_dir: Path, file_name: str):
    frames = []
    for path_text in sorted(glob.glob(str(output_dir / "seed_*" / file_name))):
        path = Path(path_text)
        seed_text = path.parent.name.replace("seed_", "")
        try:
            train_seed = int(seed_text)
        except ValueError:
            continue
        frames.append(add_train_seed_column(pd.read_csv(path), train_seed))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def flatten_columns(dataframe):
    dataframe = dataframe.copy()
    dataframe.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in dataframe.columns
    ]
    return dataframe.reset_index()


def mean_std_summary(dataframe, group_columns: list[str], metric_columns: list[str]):
    if dataframe.empty:
        return dataframe
    available_metrics = [column for column in metric_columns if column in dataframe.columns]
    if not available_metrics:
        return pd.DataFrame()
    summary = dataframe.groupby(group_columns)[available_metrics].agg(["mean", "std"])
    return flatten_columns(summary)


def collect_all_results(output_dir: Path, include_wilcoxon: bool = False) -> None:
    metric_columns = [
        "mae",
        "rmse",
        "me",
        "r2",
        "upr",
        "slope",
        "intercept",
        "spearman",
        "n",
    ]

    overall_df = aggregate_metric_file(output_dir, "overall_metrics.csv")
    category_df = aggregate_metric_file(output_dir, "category_metrics.csv")
    risk_level_df = aggregate_metric_file(output_dir, "risk_level_metrics.csv")
    wilcoxon_df = aggregate_metric_file(output_dir, "wilcoxon_signed_rank.csv") if include_wilcoxon else pd.DataFrame()

    if not overall_df.empty:
        overall_df.to_csv(output_dir / "summary_overall_by_seed.csv", index=False)
        mean_std_summary(overall_df, ["model", "model_label"], metric_columns).to_csv(
            output_dir / "summary_overall_mean_std.csv",
            index=False,
        )

    if not category_df.empty:
        category_df.to_csv(output_dir / "summary_category_by_seed.csv", index=False)
        mean_std_summary(category_df, ["category", "model", "model_label"], metric_columns).to_csv(
            output_dir / "summary_category_mean_std.csv",
            index=False,
        )

    if not risk_level_df.empty:
        risk_level_df.to_csv(output_dir / "summary_risk_level_by_seed.csv", index=False)
        mean_std_summary(risk_level_df, ["risk_level", "model", "model_label"], metric_columns).to_csv(
            output_dir / "summary_risk_level_mean_std.csv",
            index=False,
        )

    if not wilcoxon_df.empty:
        wilcoxon_df.to_csv(output_dir / "summary_wilcoxon_by_seed.csv", index=False)

    print(f"Saved aggregate summaries under {output_dir}")


def prepare_eval_dataframe(args):
    csv_path = args.csv_path if args.csv_path is not None else find_default_csv(args.data_dir)
    image_dir = args.image_dir if args.image_dir is not None else args.data_dir / "layout_images"

    if not csv_path.exists():
        raise FileNotFoundError(f"Evaluation CSV not found: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    eval_df = pd.read_csv(csv_path)
    if args.category_column not in eval_df.columns:
        raise ValueError(f"Evaluation CSV is missing category column: {args.category_column}")

    category_order = infer_category_order(eval_df, args.category_column, args.category_order)
    validate_eval_dataset(eval_df, category_order, args)

    eval_df = eval_df.copy().reset_index(drop=True)
    eval_df[args.category_column] = eval_df[args.category_column].astype(str)
    eval_df["resolved_image_path"] = eval_df[args.image_column].apply(
        lambda value: str(resolve_image_path(value, image_dir))
    )

    return eval_df, category_order, csv_path, image_dir


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate multi-seed models on the additional C0-C4 dataset."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/additional_evaluation_dataset"))
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/multi_seed_training"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/additional_evaluation"))
    parser.add_argument("--models", choices=["resnet18", "vit_tiny", "all"], default="all")

    parser.add_argument("--category-column", default="category")
    parser.add_argument("--image-column", default="layout_image_path")
    parser.add_argument("--target-column", default="invaded_grid_count")
    parser.add_argument("--category-order", type=parse_str_list, default=None)

    parser.add_argument("--train-seeds", type=parse_int_list, default=parse_int_list("42,43,44"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default=None)

    parser.add_argument("--expected-total", type=int, default=1000)
    parser.add_argument("--expected-per-category", type=int, default=200)
    parser.add_argument("--no-strict-counts", action="store_false", dest="strict_counts")
    parser.add_argument("--no-cache-images", action="store_false", dest="cache_images")
    parser.add_argument("--run-wilcoxon", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(strict_counts=True, cache_images=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    model_names = ["resnet18", "vit_tiny"] if args.models == "all" else [args.models]

    load_libraries(require_timm="vit_tiny" in model_names, require_wilcoxon=args.run_wilcoxon)
    set_seed(42)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    eval_df, category_order, csv_path, image_dir = prepare_eval_dataframe(args)
    print(f"Using device: {device}")
    print(f"Evaluation CSV: {csv_path}")
    print(f"Images: {image_dir}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Categories: {', '.join(category_order)}")

    cache_path = args.output_dir / "cache" / "additional_eval_images.npy" if args.cache_images else None
    images = load_or_build_image_cache(eval_df, "resolved_image_path", cache_path)
    transforms_pair = make_preprocess_transform(args.resize_size)

    save_json(
        args.output_dir / "evaluation_config.json",
        {
            "csv_path": str(csv_path),
            "image_dir": str(image_dir),
            "model_dir": str(args.model_dir),
            "models": model_names,
            "train_seeds": [int(seed) for seed in args.train_seeds],
            "category_order": category_order,
            "batch_size": int(args.batch_size),
            "resize_size": int(args.resize_size),
            "strict_counts": bool(args.strict_counts),
            "run_wilcoxon": bool(args.run_wilcoxon),
        },
    )

    for train_seed in args.train_seeds:
        evaluate_seed(
            train_seed,
            model_names,
            eval_df,
            images,
            category_order,
            args,
            device,
            transforms_pair,
        )

    collect_all_results(args.output_dir, include_wilcoxon=args.run_wilcoxon)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
