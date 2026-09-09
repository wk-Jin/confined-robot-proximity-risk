"""
Create selected paper figures from saved experiment outputs.

This script reproduces the paper figures generated from the additional analysis
and Score-CAM notebooks:

1. Figure 5: target range distribution for clean-success, excluded, and additional
   evaluation samples.
2. Figure 6: category-level MAE, R2, and underprediction rate.
3. Figure 7: category-level prediction-response slope and Spearman correlation.
4. Figure 8: cumulative screening gain curve averaged across training seeds.
5. Figure 9: risk-level MAE and mean error.
6. Figure 10: combined fence/pillar clean-success and risk summary.
7. Score-CAM: representative prediction cases and model attention overlays.

The figures are generated from CSV outputs, saved predictions, and model
checkpoints rather than hard-coded metric values. Additional paper figures can
be added as separate functions while keeping this script as the single entry
point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


np = None
pd = None
mpl = None
plt = None
make_axes_locatable = None
cv2 = None
torch = None
nn = None
models = None
timm = None


FIGURE_CHOICES = [
    "target_distribution",
    "category_summary",
    "trend_summary",
    "screening_gain",
    "risk_level_summary",
    "fence_pillar",
    "scorecam",
]
MODEL_COLUMN_ALIASES = {
    "resnet18": ["resnet18_pred", "cnn_pred"],
    "vit_tiny": ["vit_tiny_pred", "vit_pred"],
}
ERROR_COLUMN_ALIASES = {
    "resnet18": ["resnet18_error", "cnn_error"],
    "vit_tiny": ["vit_tiny_error", "vit_error"],
}
ABS_ERROR_COLUMN_ALIASES = {
    "resnet18": ["resnet18_abs_error", "cnn_abs_error"],
    "vit_tiny": ["vit_tiny_abs_error", "vit_abs_error"],
}
MODEL_LABELS = {
    "resnet18": "ResNet-18",
    "vit_tiny": "ViT-Tiny",
}
MODEL_NAME_ALIASES = {
    "cnn": "resnet18",
    "resnet": "resnet18",
    "resnet18": "resnet18",
    "resnet-18": "resnet18",
    "vit": "vit_tiny",
    "vit_tiny": "vit_tiny",
    "vit-tiny": "vit_tiny",
    "vittiny": "vit_tiny",
}
MODEL_FILE_NAMES = {
    "resnet18": ["resnet18_layout_risk.pt"],
    "vit_tiny": ["vit_tiny_layout_risk.pt", "vit_layout_risk.pt"],
}
CATEGORY_ORDER_PREFERRED = [
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
RISK_ORDER = ["Low", "Medium", "High", "Extreme"]
SUMMARY_METRICS = ["mae", "rmse", "me", "r2", "upr", "slope", "intercept", "spearman"]
LINE_COLOR_RESNET = "#1f77b4"
LINE_COLOR_VIT = "#d62728"
BAND_ALPHA_RESNET = 0.10
BAND_ALPHA_VIT = 0.10
LINE_WIDTH = 2.2
BAND_EDGE_LINEWIDTH = 0.8


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one integer seed.")
    return values


def parse_str_list(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one value.")
    return values


def parse_float_pair(value: str) -> tuple[float, float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(values) != 2:
        raise argparse.ArgumentTypeError("Provide exactly two comma-separated numbers.")
    return values[0], values[1]


def parse_figures(value: str) -> list[str]:
    values = parse_str_list(value)
    if values == ["all"] or "all" in values:
        return list(FIGURE_CHOICES)
    invalid = [item for item in values if item not in FIGURE_CHOICES]
    if invalid:
        choices = ", ".join(["all"] + FIGURE_CHOICES)
        raise argparse.ArgumentTypeError(f"Invalid figure(s): {invalid}. Choices: {choices}")
    return values


def load_plot_libraries() -> None:
    """Import plotting libraries after CLI parsing so --help stays lightweight."""
    global np, pd, mpl, plt, make_axes_locatable

    try:
        import numpy as np_module
        import pandas as pd_module
        import matplotlib as mpl_module

        mpl_module.use("Agg")
        import matplotlib.pyplot as plt_module
        from mpl_toolkits.axes_grid1 import make_axes_locatable as locator_function
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install the project requirements "
            "before generating paper figures."
        ) from exc

    np = np_module
    pd = pd_module
    mpl = mpl_module
    plt = plt_module
    make_axes_locatable = locator_function


def load_scorecam_libraries() -> None:
    """Import Score-CAM dependencies only when the Score-CAM figure is requested."""
    global cv2, torch, nn, models, timm

    try:
        import cv2 as cv2_module
        import torch as torch_module
        from torch import nn as nn_module
        from torchvision import models as models_module
        import timm as timm_module
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install torch, torchvision, timm, "
            "and opencv-python before generating Score-CAM figures."
        ) from exc

    cv2 = cv2_module
    torch = torch_module
    nn = nn_module
    models = models_module
    timm = timm_module


def configure_style() -> None:
    mpl.rcParams["font.family"] = "DejaVu Sans"
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["axes.linewidth"] = 1.2


def save_figure(fig, output_dir: Path, name: str, formats: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        if fmt.lower() == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path}")
    plt.close(fig)


def style_axis(ax, tick_labelsize: int = 8, grid: bool = True) -> None:
    ax.tick_params(labelsize=tick_labelsize)
    if grid:
        ax.grid(True, linewidth=0.55, alpha=0.35)


def clean_success_mask(series):
    if series.dtype == bool:
        return series

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return numeric.fillna(0).astype(int) == 1

    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin(["1", "true", "yes", "y"])


def find_default_eval_csv(eval_data_dir: Path) -> Path:
    candidates = [
        eval_data_dir / "additional_evaluation_dataset.csv",
        eval_data_dir / "phase2_trial_summary.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_seed_prediction_path(evaluation_dir: Path, train_seed: int) -> Path:
    candidates = [
        evaluation_dir / f"seed_{train_seed}" / "all_predictions.csv",
        evaluation_dir / f"seed{train_seed}" / "all_predictions.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def canonicalize_prediction_columns(dataframe):
    dataframe = dataframe.copy()
    for model_name, aliases in MODEL_COLUMN_ALIASES.items():
        canonical = aliases[0]
        if canonical in dataframe.columns:
            continue
        for alias in aliases[1:]:
            if alias in dataframe.columns:
                dataframe[canonical] = dataframe[alias]
                break
    for model_name, aliases in ERROR_COLUMN_ALIASES.items():
        canonical = aliases[0]
        if canonical in dataframe.columns:
            continue
        for alias in aliases[1:]:
            if alias in dataframe.columns:
                dataframe[canonical] = dataframe[alias]
                break
    for model_name, aliases in ABS_ERROR_COLUMN_ALIASES.items():
        canonical = aliases[0]
        if canonical in dataframe.columns:
            continue
        for alias in aliases[1:]:
            if alias in dataframe.columns:
                dataframe[canonical] = dataframe[alias]
                break
        pred_col = MODEL_COLUMN_ALIASES[model_name][0]
        err_col = ERROR_COLUMN_ALIASES[model_name][0]
        if canonical not in dataframe.columns and err_col in dataframe.columns:
            dataframe[canonical] = dataframe[err_col].abs()
        if err_col not in dataframe.columns and pred_col in dataframe.columns and "true_value" in dataframe.columns:
            dataframe[err_col] = dataframe[pred_col] - dataframe["true_value"]
            dataframe[canonical] = dataframe[err_col].abs()
    return dataframe


def load_seed_predictions(evaluation_dir: Path, train_seeds: list[int]):
    seed_dfs = {}
    for seed in train_seeds:
        path = find_seed_prediction_path(evaluation_dir, seed)
        if not path.exists():
            raise FileNotFoundError(f"Prediction CSV not found for seed {seed}: {path}")
        dataframe = canonicalize_prediction_columns(pd.read_csv(path))
        if "true_value" not in dataframe.columns:
            raise ValueError(f"Prediction CSV must contain true_value: {path}")
        for column in ["resnet18_pred", "vit_tiny_pred"]:
            if column not in dataframe.columns:
                raise ValueError(f"Prediction CSV is missing {column}: {path}")
        seed_dfs[seed] = dataframe
    return seed_dfs


def load_additional_true_values(args, seed_dfs=None):
    if seed_dfs:
        first_seed = sorted(seed_dfs.keys())[0]
        return seed_dfs[first_seed]["true_value"].astype(float)

    eval_csv = args.additional_eval_csv
    if eval_csv is None:
        eval_csv = find_default_eval_csv(args.additional_eval_data_dir)
    if not eval_csv.exists():
        raise FileNotFoundError(f"Additional evaluation CSV not found: {eval_csv}")
    dataframe = pd.read_csv(eval_csv)
    if args.target_column not in dataframe.columns:
        raise ValueError(f"Evaluation CSV is missing {args.target_column}: {eval_csv}")
    return dataframe[args.target_column].astype(float)


def load_training_dataframe(path: Path, target_column: str, clean_column: str, required_extra=None):
    if not path.exists():
        raise FileNotFoundError(f"Training CSV not found: {path}")

    required = [target_column, clean_column]
    if required_extra:
        required.extend(required_extra)
    dataframe = pd.read_csv(path, usecols=lambda column: column in set(required))

    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Training CSV is missing required columns: {missing}")
    return dataframe


def plot_target_distribution(args, seed_dfs=None) -> None:
    colors = {
        "blue": "#2a78d6",
        "red": "#e34948",
        "muted": "#8c8c8c",
    }
    dataframe = load_training_dataframe(
        args.training_csv,
        args.target_column,
        args.clean_column,
    )
    mask = clean_success_mask(dataframe[args.clean_column])
    clean_success = dataframe.loc[mask, args.target_column].astype(float)
    non_clean_success = dataframe.loc[~mask, args.target_column].astype(float)
    additional_values = load_additional_true_values(args, seed_dfs)
    clean_max = clean_success.max()

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.95, bottom=0.14)

    ax.hist(
        clean_success,
        bins=args.histogram_bins,
        density=True,
        color=colors["blue"],
        alpha=0.5,
        label="Clean-success dataset",
    )
    ax.hist(
        non_clean_success,
        bins=args.histogram_bins,
        density=True,
        color=colors["red"],
        alpha=0.4,
        label="Excluded non-clean-success trials",
    )
    ax.hist(
        additional_values,
        bins=args.histogram_bins,
        density=True,
        histtype="step",
        color="black",
        linewidth=1.7,
        label="Additional evaluation (C0-C4)",
    )
    ax.axvline(
        clean_max,
        color=colors["muted"],
        linestyle="--",
        linewidth=0.8,
        label=f"Clean-success maximum ({clean_max:.0f})",
    )

    ax.set_xlabel(args.target_column, fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(frameon=False, fontsize=10)
    style_axis(ax, tick_labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)

    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "group": (
                ["clean_success"] * len(clean_success)
                + ["non_clean_success"] * len(non_clean_success)
                + ["additional_evaluation"] * len(additional_values)
            ),
            args.target_column: (
                clean_success.to_list()
                + non_clean_success.to_list()
                + additional_values.to_list()
            ),
        }
    ).to_csv(source_dir / "target_distribution_values.csv", index=False)

    save_figure(fig, args.output_dir, "fig_5_target_range_distribution", args.formats)


def cumulative_gain_curve(dataframe, pred_col: str, true_col: str = "true_value", positive_quantile: float = 0.75, n_points: int = 50):
    n_rows = len(dataframe)
    if n_rows == 0:
        raise ValueError("Cannot compute a gain curve from an empty dataframe.")

    threshold = dataframe[true_col].quantile(positive_quantile)
    is_positive = (dataframe[true_col] >= threshold).to_numpy()
    n_positive = int(is_positive.sum())
    if n_positive == 0:
        raise ValueError("No positive samples found for the requested quantile.")

    order = np.argsort(-dataframe[pred_col].to_numpy())
    sorted_positive = is_positive[order]

    ks = np.linspace(0.0, 1.0, n_points)
    recalls = np.array(
        [
            sorted_positive[: max(1, int(np.ceil(k * n_rows)))].sum() / n_positive
            for k in ks
        ]
    )
    return ks, recalls


def gain_curves_all_seeds(seed_dfs, pred_col: str, positive_quantile: float, n_points: int):
    ks_ref = None
    all_recalls = []
    for dataframe in seed_dfs.values():
        ks_ref, recalls = cumulative_gain_curve(
            dataframe,
            pred_col,
            positive_quantile=positive_quantile,
            n_points=n_points,
        )
        all_recalls.append(recalls)
    all_recalls = np.asarray(all_recalls)
    return ks_ref, all_recalls.mean(axis=0), all_recalls.std(axis=0)


def plot_screening_gain(args, seed_dfs) -> None:
    line_color_resnet = "#1f77b4"
    line_color_vit = "#d62728"
    muted = "#8c8c8c"
    band_alpha = 0.13

    ks, resnet_mean, resnet_std = gain_curves_all_seeds(
        seed_dfs,
        "resnet18_pred",
        positive_quantile=args.positive_quantile,
        n_points=args.gain_points,
    )
    _, vit_mean, vit_std = gain_curves_all_seeds(
        seed_dfs,
        "vit_tiny_pred",
        positive_quantile=args.positive_quantile,
        n_points=args.gain_points,
    )

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    fig.subplots_adjust(left=0.13, right=0.96, top=0.95, bottom=0.13)

    ax.plot([0, 1], [0, 1], color=muted, linestyle=":", linewidth=0.8, label="Random")
    ax.plot(ks, resnet_mean, color=line_color_resnet, linestyle="-", linewidth=1.7, label=MODEL_LABELS["resnet18"])
    ax.fill_between(ks, resnet_mean - resnet_std, resnet_mean + resnet_std, color=line_color_resnet, alpha=band_alpha, linewidth=0.0)
    ax.plot(ks, vit_mean, color=line_color_vit, linestyle="--", linewidth=1.7, label=MODEL_LABELS["vit_tiny"])
    ax.fill_between(ks, vit_mean - vit_std, vit_mean + vit_std, color=line_color_vit, alpha=band_alpha, linewidth=0.0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    ax.set_xlabel("Top-K screened fraction", fontsize=12)
    ax.set_ylabel("Recall of target samples", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    style_axis(ax, tick_labelsize=10)

    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "screened_fraction": ks,
            "random": ks,
            "resnet18_mean": resnet_mean,
            "resnet18_std": resnet_std,
            "vit_tiny_mean": vit_mean,
            "vit_tiny_std": vit_std,
        }
    ).to_csv(source_dir / "screening_gain_curve.csv", index=False)

    save_figure(fig, args.output_dir, "fig_8_screening_gain", args.formats)


def grouped_count(dataframe, columns):
    return dataframe.groupby(columns, dropna=False).size().sort_index()


def plot_fence_pillar_combined(args) -> None:
    required_extra = [
        args.fence_column,
        args.pillar_x_column,
        args.pillar_y_column,
    ]
    dataframe = load_training_dataframe(
        args.training_csv,
        args.target_column,
        args.clean_column,
        required_extra=required_extra,
    )
    mask = clean_success_mask(dataframe[args.clean_column])
    train_clean = dataframe.loc[mask].copy()

    blue = "#2a78d6"
    red = "#e34948"
    band_alpha = 0.13

    fig_width = 7.2
    fence_row_height = 2.0
    pillar_row_height = 2.0
    hspace_in = 0.8
    top_margin_in = 0.20
    bottom_margin_in = 0.62
    fig_height = top_margin_in + fence_row_height + hspace_in + pillar_row_height + bottom_margin_in
    top_frac = 1 - top_margin_in / fig_height
    bottom_frac = bottom_margin_in / fig_height
    hspace_frac = hspace_in / ((fence_row_height + pillar_row_height) / 2)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": [fence_row_height, pillar_row_height]},
    )
    fig.subplots_adjust(left=0.10, right=0.89, top=top_frac, bottom=bottom_frac, wspace=0.5, hspace=hspace_frac)
    ax_fence_surv, ax_fence_risk = axes[0]
    ax_pillar_surv, ax_pillar_risk = axes[1]

    raw_fence = grouped_count(dataframe, [args.fence_column])
    clean_fence = grouped_count(train_clean, [args.fence_column]).reindex(raw_fence.index, fill_value=0)
    fence_rate = clean_fence / raw_fence * 100.0
    fence_x_sorted = list(fence_rate.index)

    ax_fence_surv.plot(fence_rate.index, fence_rate.values, color=blue, marker="o", markersize=4.4, linewidth=1.7)
    ax_fence_surv.set_ylim(0, 100)
    ax_fence_surv.set_xticks(fence_x_sorted)
    ax_fence_surv.set_xticklabels([f"{float(value):.3f}" for value in fence_x_sorted], rotation=45, fontsize=8)
    ax_fence_surv.set_xlabel(args.fence_column, fontsize=8)
    ax_fence_surv.set_ylabel("Clean-success rate (%)", fontsize=8)
    style_axis(ax_fence_surv)

    fence_group = train_clean.groupby(args.fence_column)[args.target_column]
    fence_mean = fence_group.mean().sort_index()
    fence_std = fence_group.std().fillna(0.0).reindex(fence_mean.index)

    ax_fence_risk.plot(fence_mean.index, fence_mean.values, color=red, marker="o", markersize=4.4, linewidth=1.7)
    ax_fence_risk.fill_between(fence_mean.index, fence_mean - fence_std, fence_mean + fence_std, color=red, alpha=band_alpha, linewidth=0.0)
    lo, hi = float((fence_mean - fence_std).min()), float((fence_mean + fence_std).max())
    pad = (hi - lo) * 0.10 if hi > lo else 1.0
    ax_fence_risk.set_ylim(lo - pad, hi + pad)
    ax_fence_risk.set_xticks(fence_x_sorted)
    ax_fence_risk.set_xticklabels([f"{float(value):.3f}" for value in fence_x_sorted], rotation=45, fontsize=8)
    ax_fence_risk.set_xlabel(args.fence_column, fontsize=8)
    ax_fence_risk.set_ylabel(f"Mean {args.target_column}", fontsize=8)
    style_axis(ax_fence_risk)

    pillar_columns = [args.pillar_y_column, args.pillar_x_column]
    pillar_raw = dataframe.groupby(pillar_columns).size().unstack().sort_index().sort_index(axis=1)
    pillar_clean = train_clean.groupby(pillar_columns).size().unstack().reindex_like(pillar_raw).fillna(0.0)
    pillar_rate = pillar_clean / pillar_raw * 100.0

    im1 = ax_pillar_surv.imshow(pillar_rate.values, origin="lower", cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax_pillar_surv.set_xlabel(args.pillar_x_column, fontsize=8)
    ax_pillar_surv.set_xticks(range(len(pillar_rate.columns)))
    ax_pillar_surv.set_xticklabels([f"{float(value):.2f}" for value in pillar_rate.columns], fontsize=8)
    ax_pillar_surv.set_yticks(range(len(pillar_rate.index)))
    ax_pillar_surv.set_yticklabels([f"{float(value):.2f}" for value in pillar_rate.index], fontsize=8)
    ax_pillar_surv.yaxis.tick_right()
    ax_pillar_surv.yaxis.set_label_position("right")
    ax_pillar_surv.set_ylabel(args.pillar_y_column, fontsize=8)

    div1 = make_axes_locatable(ax_pillar_surv)
    cax1 = div1.append_axes("left", size="5%", pad=0.15)
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("Clean-success rate (%)", fontsize=8)
    cb1.ax.tick_params(labelsize=8)
    cb1.ax.yaxis.set_ticks_position("left")
    cb1.ax.yaxis.set_label_position("left")

    risk_group = train_clean.groupby(pillar_columns)[args.target_column]
    risk_mean = risk_group.mean().unstack().sort_index().sort_index(axis=1)
    risk_count = risk_group.size().unstack().reindex_like(risk_mean)
    risk_masked = risk_mean.mask(risk_count < args.min_risk_count)

    im2 = ax_pillar_risk.imshow(risk_masked.values, origin="lower", cmap="Reds", aspect="auto")
    ax_pillar_risk.set_xlabel(args.pillar_x_column, fontsize=8)
    ax_pillar_risk.set_xticks(range(len(risk_masked.columns)))
    ax_pillar_risk.set_xticklabels([f"{float(value):.2f}" for value in risk_masked.columns], fontsize=8)
    ax_pillar_risk.set_yticks(range(len(risk_masked.index)))
    ax_pillar_risk.set_yticklabels([f"{float(value):.2f}" for value in risk_masked.index], fontsize=8)
    ax_pillar_risk.yaxis.tick_right()
    ax_pillar_risk.yaxis.set_label_position("right")
    ax_pillar_risk.set_ylabel(args.pillar_y_column, fontsize=8)

    div2 = make_axes_locatable(ax_pillar_risk)
    cax2 = div2.append_axes("left", size="5%", pad=0.15)
    cb2 = fig.colorbar(im2, cax=cax2)
    cb2.set_label(f"Mean {args.target_column}", fontsize=8)
    cb2.ax.tick_params(labelsize=8)
    cb2.ax.yaxis.set_ticks_position("left")
    cb2.ax.yaxis.set_label_position("left")

    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            args.fence_column: fence_rate.index,
            "raw_n": raw_fence.values,
            "clean_n": clean_fence.values,
            "clean_success_rate": fence_rate.values,
        }
    ).to_csv(source_dir / "fence_clean_success_rate.csv", index=False)
    pd.DataFrame(
        {
            args.fence_column: fence_mean.index,
            "n": fence_group.size().sort_index().values,
            "mean": fence_mean.values,
            "std": fence_std.values,
        }
    ).to_csv(source_dir / "fence_risk_summary.csv", index=False)

    pillar_rate.stack().rename("clean_success_rate").reset_index().to_csv(
        source_dir / "pillar_clean_success_rate.csv",
        index=False,
    )
    risk_source = risk_mean.stack().rename("mean").reset_index()
    risk_source["n"] = risk_count.stack().to_numpy()
    risk_source.to_csv(source_dir / "pillar_risk_summary.csv", index=False)

    save_figure(fig, args.output_dir, "fig_10_fence_pillar_combined", args.formats)


def category_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    if text.startswith("C") and len(text) > 1 and text[1].isdigit():
        return int(text[1]), text
    return 99, text


def short_category_label(value: str) -> str:
    text = str(value).strip()
    if "_" in text:
        text = text.split("_", 1)[0]
    if " " in text:
        text = text.split(" ", 1)[0]
    if "(" in text:
        text = text.split("(", 1)[0]
    return text.strip()


def infer_order(observed_values, preferred_values) -> list[str]:
    observed = [str(value) for value in observed_values if str(value) != "nan"]
    preferred = [value for value in preferred_values if value in observed]
    remainder = [value for value in observed if value not in preferred]
    return preferred + sorted(remainder, key=category_sort_key)


def find_metric_summary_path(evaluation_dir: Path, explicit_path: Path | None, candidates: list[str]) -> Path:
    if explicit_path is not None:
        return explicit_path
    for file_name in candidates:
        candidate = evaluation_dir / file_name
        if candidate.exists():
            return candidate
    return evaluation_dir / candidates[0]


def wide_metrics_to_long(dataframe, group_column: str):
    rows = []
    prefix_map = {
        "cnn": "resnet18",
        "resnet18": "resnet18",
        "vit": "vit_tiny",
        "vit_tiny": "vit_tiny",
    }
    for _, row in dataframe.iterrows():
        for prefix, model_name in prefix_map.items():
            metric_values = {
                metric: row[f"{prefix}_{metric}"]
                for metric in SUMMARY_METRICS
                if f"{prefix}_{metric}" in dataframe.columns
            }
            if not metric_values:
                continue
            out = {
                group_column: row[group_column],
                "model": model_name,
                "model_label": MODEL_LABELS[model_name],
            }
            for optional_column in ["train_seed", "n"]:
                if optional_column in dataframe.columns:
                    out[optional_column] = row[optional_column]
            out.update(metric_values)
            rows.append(out)
    return pd.DataFrame(rows)


def load_metric_summary(path: Path, group_column: str):
    if not path.exists():
        raise FileNotFoundError(f"Metric summary CSV not found: {path}")
    dataframe = pd.read_csv(path)
    if group_column not in dataframe.columns:
        raise ValueError(f"Metric summary CSV is missing {group_column}: {path}")

    if "model" not in dataframe.columns:
        dataframe = wide_metrics_to_long(dataframe, group_column)
    else:
        dataframe = dataframe.copy()
        dataframe["model"] = dataframe["model"].apply(canonical_model_name)
        if "model_label" not in dataframe.columns:
            dataframe["model_label"] = dataframe["model"].map(MODEL_LABELS)

    if dataframe.empty:
        raise ValueError(f"No plottable metrics found in {path}")
    return dataframe


def load_category_metrics(args):
    path = find_metric_summary_path(
        args.evaluation_dir,
        args.category_metrics_csv,
        ["summary_category_by_seed.csv", "summary_table_4_3.csv"],
    )
    return load_metric_summary(path, args.category_column), path


def load_risk_level_metrics(args):
    path = find_metric_summary_path(
        args.evaluation_dir,
        args.risk_level_metrics_csv,
        ["summary_risk_level_by_seed.csv", "summary_table_4_4.csv"],
    )
    return load_metric_summary(path, "risk_level"), path


def summarize_metric(dataframe, group_column: str, order: list[str], model_name: str, metric: str):
    subset = dataframe[dataframe["model"] == model_name]
    if metric not in subset.columns:
        raise ValueError(f"Metric summary is missing metric column: {metric}")
    stat = subset.groupby(group_column)[metric].agg(["mean", "std"]).reindex(order)
    stat["std"] = stat["std"].fillna(0.0)
    return stat


def plot_trend_summary(args) -> None:
    category_df, source_path = load_category_metrics(args)
    cat_order = infer_order(category_df[args.category_column].dropna().unique(), CATEGORY_ORDER_PREFERRED)
    x = np.arange(len(cat_order))
    x_labels = [short_category_label(category) for category in cat_order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="white")
    metric_specs = [
        ("slope", "Prediction-response slope", axes[0]),
        ("spearman", "Spearman correlation", axes[1]),
    ]
    model_specs = [
        ("resnet18", MODEL_LABELS["resnet18"], LINE_COLOR_RESNET, BAND_ALPHA_RESNET),
        ("vit_tiny", MODEL_LABELS["vit_tiny"], LINE_COLOR_VIT, BAND_ALPHA_VIT),
    ]

    legend_handles = []
    summary_rows = []
    for metric_key, metric_label, ax in metric_specs:
        for model_name, model_label, color, band_alpha in model_specs:
            stat = summarize_metric(category_df, args.category_column, cat_order, model_name, metric_key)
            mean_vals = stat["mean"].to_numpy(dtype=float)
            std_vals = stat["std"].to_numpy(dtype=float)
            line, = ax.plot(
                x,
                mean_vals,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=5.5,
                color=color,
                label=f"{model_label} (mean +/- std)",
                zorder=4,
            )
            ax.fill_between(
                x,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=band_alpha,
                edgecolor=color,
                linewidth=BAND_EDGE_LINEWIDTH,
                zorder=2,
            )
            if metric_key == "slope":
                legend_handles.append(line)
            for category, mean_value, std_value in zip(cat_order, mean_vals, std_vals):
                summary_rows.append(
                    {
                        "figure": "fig_7_trend_summary_combined",
                        "source_csv": str(source_path),
                        "category": category,
                        "model": model_name,
                        "metric": metric_key,
                        "mean": mean_value,
                        "std": std_value,
                    }
                )

        if metric_key == "spearman":
            ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Extrapolation category", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=10)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=10,
        markerscale=1.5,
    )
    plt.tight_layout()

    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(source_dir / "fig_7_trend_summary_combined.csv", index=False)
    save_figure(fig, args.output_dir, "fig_7_trend_summary_combined", args.formats)


def plot_risk_level_summary(args) -> None:
    risk_df, source_path = load_risk_level_metrics(args)
    risk_order = infer_order(risk_df["risk_level"].dropna().unique(), RISK_ORDER)
    risk_n = risk_df.groupby("risk_level")["n"].mean().round().astype(int).reindex(risk_order)
    x_labels = [f"{level}\n(n={int(risk_n.loc[level])})" if not pd.isna(risk_n.loc[level]) else level for level in risk_order]
    x = np.arange(len(risk_order))
    bar_width = 0.32

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="white")
    metric_specs = [
        ("mae", "(a) MAE by risk level", "MAE", axes[0]),
        ("me", "(b) Mean error by risk level", "ME (+ over-, - under-prediction)", axes[1]),
    ]
    model_specs = [
        ("resnet18", MODEL_LABELS["resnet18"], LINE_COLOR_RESNET),
        ("vit_tiny", MODEL_LABELS["vit_tiny"], LINE_COLOR_VIT),
    ]

    legend_handles, legend_labels = [], []
    summary_rows = []
    for metric_key, metric_title, ylabel, ax in metric_specs:
        for index, (model_name, model_label, color) in enumerate(model_specs):
            stat = summarize_metric(risk_df, "risk_level", risk_order, model_name, metric_key)
            offset = (index - 0.5) * bar_width
            bars = ax.bar(
                x + offset,
                stat["mean"],
                width=bar_width,
                yerr=stat["std"].fillna(0.0),
                color=color,
                capsize=3,
                label=model_label,
                error_kw={"elinewidth": 1.1, "ecolor": "black", "capthick": 1.1},
            )
            if metric_key == "mae":
                legend_handles.append(bars)
                legend_labels.append(model_label)
            for level, mean_value, std_value in zip(risk_order, stat["mean"], stat["std"]):
                summary_rows.append(
                    {
                        "figure": "fig_9_risk_level_summary_mae_me",
                        "source_csv": str(source_path),
                        "risk_level": level,
                        "model": model_name,
                        "metric": metric_key,
                        "mean": mean_value,
                        "std": std_value,
                    }
                )

        if metric_key == "me":
            ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(metric_title, fontsize=12, fontweight="bold", loc="left")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(alpha=0.25, axis="y")
        ax.tick_params(labelsize=9)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout()

    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(source_dir / "fig_9_risk_level_summary_mae_me.csv", index=False)
    save_figure(fig, args.output_dir, "fig_9_risk_level_summary_mae_me", args.formats)


def plot_category_summary(args) -> None:
    category_df, source_path = load_category_metrics(args)
    cat_order = infer_order(category_df[args.category_column].dropna().unique(), CATEGORY_ORDER_PREFERRED)
    x = np.arange(len(cat_order))
    x_labels = [short_category_label(category) for category in cat_order]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), facecolor="white")
    metric_specs = [
        ("mae", "(a) MAE by category", "MAE", axes[0]),
        ("r2", "(b) R2 by category", "R2", axes[1]),
        ("upr", "(c) Underprediction rate by category", "UPR (%)", axes[2]),
    ]
    model_specs = [
        ("resnet18", MODEL_LABELS["resnet18"], LINE_COLOR_RESNET, BAND_ALPHA_RESNET),
        ("vit_tiny", MODEL_LABELS["vit_tiny"], LINE_COLOR_VIT, BAND_ALPHA_VIT),
    ]

    legend_handles, legend_labels = [], []
    summary_rows = []
    for metric_key, metric_title, ylabel, ax in metric_specs:
        for model_name, model_label, color, band_alpha in model_specs:
            stat = summarize_metric(category_df, args.category_column, cat_order, model_name, metric_key)
            mean_vals = stat["mean"].to_numpy(dtype=float)
            std_vals = stat["std"].to_numpy(dtype=float)
            line, = ax.plot(
                x,
                mean_vals,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=5.5,
                color=color,
                label=model_label,
                zorder=4,
            )
            ax.fill_between(
                x,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=band_alpha,
                edgecolor=color,
                linewidth=BAND_EDGE_LINEWIDTH,
                zorder=2,
            )
            if metric_key == "r2":
                ax.axhline(0, color="gray", linewidth=0.9, alpha=0.6)
            if metric_key == "mae":
                legend_handles.append(line)
                legend_labels.append(model_label)
            for category, mean_value, std_value in zip(cat_order, mean_vals, std_vals):
                summary_rows.append(
                    {
                        "figure": "fig_6_category_summary_mae_r2_upr",
                        "source_csv": str(source_path),
                        "category": category,
                        "model": model_name,
                        "metric": metric_key,
                        "mean": mean_value,
                        "std": std_value,
                    }
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Extrapolation category", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(metric_title, fontsize=12, fontweight="bold", loc="left")
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=9)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=10,
    )
    plt.tight_layout()

    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(source_dir / "fig_6_category_summary_mae_r2_upr.csv", index=False)
    save_figure(fig, args.output_dir, "fig_6_category_summary_mae_r2_upr", args.formats)


def canonical_model_name(value: str) -> str:
    key = str(value).strip().lower().replace(" ", "").replace("_", "-")
    if key not in MODEL_NAME_ALIASES:
        raise ValueError(f"Unsupported model name in metrics file: {value}")
    return MODEL_NAME_ALIASES[key]


def find_default_overall_metrics_path(evaluation_dir: Path) -> Path | None:
    candidates = [
        evaluation_dir / "summary_overall_by_seed.csv",
        evaluation_dir / "summary_table_4_2.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def r2_score_np(true_values, predictions) -> float:
    true_values = np.asarray(true_values, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    ss_res = np.sum((true_values - predictions) ** 2)
    ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_scorecam_overall_metrics(seed_dfs):
    rows = []
    for train_seed, dataframe in seed_dfs.items():
        true_values = dataframe["true_value"].to_numpy(dtype=float)
        for model_name in ["resnet18", "vit_tiny"]:
            pred_col = MODEL_COLUMN_ALIASES[model_name][0]
            predictions = dataframe[pred_col].to_numpy(dtype=float)
            errors = predictions - true_values
            rows.append(
                {
                    "train_seed": int(train_seed),
                    "model": model_name,
                    "mae": float(np.mean(np.abs(errors))),
                    "rmse": float(np.sqrt(np.mean(errors**2))),
                    "me": float(np.mean(errors)),
                    "r2": r2_score_np(true_values, predictions),
                    "upr": float(np.mean(true_values > predictions) * 100.0),
                }
            )
    return pd.DataFrame(rows)


def load_scorecam_overall_metrics(args, seed_dfs):
    if args.scorecam_overall_csv is not None:
        path = args.scorecam_overall_csv
    else:
        path = find_default_overall_metrics_path(args.evaluation_dir)

    if path is None:
        return compute_scorecam_overall_metrics(seed_dfs)

    if not path.exists():
        raise FileNotFoundError(f"Overall metrics CSV not found: {path}")

    dataframe = pd.read_csv(path)
    required = ["train_seed", "model", "mae", "rmse", "r2", "me", "upr"]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Overall metrics CSV is missing required columns: {missing}")

    dataframe = dataframe.copy()
    dataframe["model"] = dataframe["model"].apply(canonical_model_name)
    return dataframe


def select_representative_seed(overall_df, train_seeds: list[int], output_dir: Path) -> int:
    metrics = ["mae", "rmse", "r2", "me", "upr"]
    overall_df = overall_df.copy()
    overall_df["model"] = overall_df["model"].apply(canonical_model_name)
    subset = overall_df[
        overall_df["train_seed"].isin(train_seeds)
        & overall_df["model"].isin(["resnet18", "vit_tiny"])
    ].copy()

    pivot = subset.pivot_table(index="train_seed", columns="model", values=metrics, aggfunc="first")
    pivot = pivot.reindex(train_seeds)
    expected_columns = [(metric, model) for metric in metrics for model in ["resnet18", "vit_tiny"]]
    missing_columns = [column for column in expected_columns if column not in pivot.columns]
    if missing_columns:
        raise ValueError(f"Cannot select representative seed; missing metrics: {missing_columns}")
    if pivot.isna().any().any():
        raise ValueError("Cannot select representative seed; some requested seeds have missing metrics.")

    pivot.columns = [f"{model}_{metric}" for metric, model in pivot.columns]
    z_scores = (pivot - pivot.mean(axis=0)) / pivot.std(axis=0, ddof=0)
    z_scores = z_scores.fillna(0.0)
    deviation_score = z_scores.abs().sum(axis=1)
    representative_seed = int(deviation_score.idxmin())

    source_dir = output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    selection_df = pivot.copy()
    selection_df["deviation_score"] = deviation_score
    selection_df["representative_seed"] = selection_df.index == representative_seed
    selection_df.to_csv(source_dir / "scorecam_representative_seed_selection.csv")
    z_scores.to_csv(source_dir / "scorecam_representative_seed_zscores.csv")

    return representative_seed


def align_scorecam_seed_predictions(seed_dfs, train_seeds: list[int]):
    aligned = {seed: dataframe.copy().reset_index(drop=True) for seed, dataframe in seed_dfs.items()}
    first_seed = train_seeds[0]
    base_true = aligned[first_seed]["true_value"].to_numpy(dtype=float)
    if all(np.allclose(aligned[seed]["true_value"].to_numpy(dtype=float), base_true) for seed in train_seeds[1:]):
        return aligned

    if all("candidate_id" in aligned[seed].columns for seed in train_seeds):
        for seed in train_seeds:
            aligned[seed] = aligned[seed].sort_values("candidate_id").reset_index(drop=True)
        base_ids = aligned[first_seed]["candidate_id"].to_numpy()
        base_true = aligned[first_seed]["true_value"].to_numpy(dtype=float)
        for seed in train_seeds[1:]:
            if not np.array_equal(aligned[seed]["candidate_id"].to_numpy(), base_ids):
                raise ValueError("Prediction CSV files do not contain the same candidate_id order.")
            if not np.allclose(aligned[seed]["true_value"].to_numpy(dtype=float), base_true):
                raise ValueError("Prediction CSV files disagree on true_value after candidate_id alignment.")
        return aligned

    raise ValueError("Prediction CSV files are not aligned and candidate_id is unavailable.")


def build_average_error_dataframe(seed_dfs, train_seeds: list[int], category_column: str):
    first = seed_dfs[train_seeds[0]]
    metadata_columns = [
        column
        for column in [
            category_column,
            "candidate_id",
            "combination_id",
            "layout_image_path",
            "resolved_image_path",
        ]
        if column in first.columns
    ]
    avg_df = first[metadata_columns].copy()
    avg_df["true_value"] = first["true_value"].astype(float).to_numpy()

    for model_name in ["resnet18", "vit_tiny"]:
        err_col = ERROR_COLUMN_ALIASES[model_name][0]
        abs_col = ABS_ERROR_COLUMN_ALIASES[model_name][0]
        avg_df[err_col] = np.mean([seed_dfs[seed][err_col].to_numpy(dtype=float) for seed in train_seeds], axis=0)
        avg_df[abs_col] = np.mean([seed_dfs[seed][abs_col].to_numpy(dtype=float) for seed in train_seeds], axis=0)

    return avg_df


def select_scorecam_cases(avg_df, category_column: str):
    dataframe = avg_df.copy()
    dataframe["combined_abs_error"] = dataframe["resnet18_abs_error"] + dataframe["vit_tiny_abs_error"]
    dataframe["divergence"] = dataframe["vit_tiny_abs_error"] - dataframe["resnet18_abs_error"]
    dataframe["combined_error"] = dataframe["resnet18_error"] + dataframe["vit_tiny_error"]

    case1 = dataframe.sort_values("combined_abs_error").iloc[0]
    case2 = dataframe.sort_values("divergence", ascending=False).iloc[0]
    threshold = dataframe["true_value"].quantile(0.75)
    high_risk = dataframe[dataframe["true_value"] >= threshold]
    case3 = high_risk.sort_values("combined_error").iloc[0]

    cases = []
    for label, row in zip(["Case 1", "Case 2", "Case 3"], [case1, case2, case3]):
        case = row.to_dict()
        case["label"] = label
        case["sample_index"] = int(row.name)
        if category_column in row.index:
            case["category"] = row[category_column]
        cases.append(case)
    return cases


def verify_scorecam_cases(cases, seed_dfs, train_seeds: list[int]):
    rows = []
    for case in cases:
        idx = int(case["sample_index"])
        row = {
            "case": case["label"],
            "category": case.get("category"),
            "true_value": float(case["true_value"]),
        }
        resnet_errors = []
        vit_errors = []
        for seed in train_seeds:
            resnet_error = float(seed_dfs[seed].loc[idx, "resnet18_error"])
            vit_error = float(seed_dfs[seed].loc[idx, "vit_tiny_error"])
            row[f"resnet18_error_seed_{seed}"] = resnet_error
            row[f"vit_tiny_error_seed_{seed}"] = vit_error
            resnet_errors.append(resnet_error)
            vit_errors.append(vit_error)

        row["resnet18_mean_error"] = float(np.mean(resnet_errors))
        row["resnet18_std_error"] = float(np.std(resnet_errors, ddof=0))
        row["vit_tiny_mean_error"] = float(np.mean(vit_errors))
        row["vit_tiny_std_error"] = float(np.std(vit_errors, ddof=0))
        row["resnet18_sign_consistent"] = len(set(np.sign(resnet_errors))) == 1
        row["vit_tiny_sign_consistent"] = len(set(np.sign(vit_errors))) == 1
        rows.append(row)
    return pd.DataFrame(rows)


def load_eval_metadata(args):
    eval_csv = args.additional_eval_csv
    if eval_csv is None:
        eval_csv = find_default_eval_csv(args.additional_eval_data_dir)
    if not eval_csv.exists():
        return None
    return pd.read_csv(eval_csv).reset_index(drop=True)


def flag_from_range(value, bounds: tuple[float, float], tolerance: float) -> bool:
    lo, hi = bounds
    return value < lo - tolerance or value > hi + tolerance


def compute_extrapolated_variables(row, args) -> str:
    if "extrap_vars" in row and pd.notna(row["extrap_vars"]):
        return str(row["extrap_vars"])

    flag_columns = [
        ("fence_extrap", "Fence"),
        ("pillar_extrap", "Pillar"),
        ("pick_extrap", "Pick"),
        ("place_extrap", "Place"),
    ]
    if all(column in row for column, _ in flag_columns):
        labels = [label for column, label in flag_columns if bool(row[column])]
        return ", ".join(labels) if labels else "None"

    required = [
        args.fence_column,
        args.pillar_x_column,
        args.pillar_y_column,
        "pick_x_offset",
        "pick_y_offset",
        "place_x_offset",
        "place_y_offset",
    ]
    if not all(column in row for column in required):
        return "Not available"

    tol = args.extrap_tolerance
    labels = []
    if flag_from_range(float(row[args.fence_column]), args.fence_range, tol):
        labels.append("Fence")
    if (
        flag_from_range(float(row[args.pillar_x_column]), args.pillar_x_range, tol)
        or flag_from_range(float(row[args.pillar_y_column]), args.pillar_y_range, tol)
    ):
        labels.append("Pillar")
    if (
        flag_from_range(float(row["pick_x_offset"]), args.pick_offset_range, tol)
        or flag_from_range(float(row["pick_y_offset"]), args.pick_offset_range, tol)
    ):
        labels.append("Pick")
    if (
        flag_from_range(float(row["place_x_offset"]), args.place_offset_range, tol)
        or flag_from_range(float(row["place_y_offset"]), args.place_offset_range, tol)
    ):
        labels.append("Place")

    return ", ".join(labels) if labels else "None"


def enrich_scorecam_cases(cases, eval_df, args):
    for case in cases:
        matched_row = None
        if eval_df is not None:
            if "candidate_id" in case and "candidate_id" in eval_df.columns:
                matches = eval_df[eval_df["candidate_id"] == case["candidate_id"]]
                if not matches.empty:
                    matched_row = matches.iloc[0]
            if matched_row is None and int(case["sample_index"]) < len(eval_df):
                matched_row = eval_df.iloc[int(case["sample_index"])]

        if matched_row is not None:
            if args.target_column in matched_row:
                true_value = float(matched_row[args.target_column])
                if abs(true_value - float(case["true_value"])) > 1e-3:
                    raise ValueError(f"Selected case does not match evaluation metadata: {case['label']}")
            for column in [args.category_column, "layout_image_path", "resolved_image_path"]:
                if column in matched_row and column not in case:
                    case[column] = matched_row[column]
            case["extrap_vars"] = compute_extrapolated_variables(matched_row, args)

        if "category" not in case and args.category_column in case:
            case["category"] = case[args.category_column]
        if "extrap_vars" not in case:
            case["extrap_vars"] = "Not available"
        if "layout_image_path" not in case and "resolved_image_path" in case:
            case["layout_image_path"] = case["resolved_image_path"]
        if "layout_image_path" not in case:
            raise ValueError(f"Selected case is missing layout_image_path: {case['label']}")
    return cases


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


def torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


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


def build_scorecam_resnet(hidden_dim: int, dropout: float):
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


def build_scorecam_vit(hidden_dim: int, dropout: float):
    model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
    in_features = model.num_features
    model.head = nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    return model


def build_scorecam_model(model_name: str, hidden_dim: int, dropout: float):
    if model_name == "resnet18":
        return build_scorecam_resnet(hidden_dim, dropout)
    if model_name == "vit_tiny":
        return build_scorecam_vit(hidden_dim, dropout)
    raise ValueError(f"Unsupported model: {model_name}")


def load_scorecam_model(model_name: str, model_path: Path, args, device):
    checkpoint = torch_load(model_path, device)
    model = build_scorecam_model(model_name, args.hidden_dim, args.dropout).to(device)
    model.load_state_dict(adapt_state_dict(checkpoint["model_state_dict"]))
    model.eval()
    return model, float(checkpoint["target_mean"]), float(checkpoint["target_std"])


def resolve_scorecam_image_path(raw_value, image_dir: Path) -> Path:
    raw_text = str(raw_value).strip()
    raw_path = Path(raw_text.replace("\\", "/"))
    candidates = []
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
    raise FileNotFoundError(f"Score-CAM image not found. Searched: {searched}")


def load_scorecam_image(path: Path):
    if path.suffix.lower() == ".npy":
        image = np.load(path)
    else:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if image_bgr.ndim == 2:
            image = image_bgr
        else:
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    if image.ndim == 3 and image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3, 4]:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def load_and_preprocess_scorecam_image(image_path: Path, device):
    raw = load_scorecam_image(image_path)
    image = raw.astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    tensor = torch.nn.functional.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor.to(device), raw


def score_cam_cnn(model, input_tensor):
    model.eval()
    activations = []

    def hook_fn(module, inputs, output):
        activations.append(output.detach())

    handle = model.layer4[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()

    if not activations:
        raise RuntimeError("No CNN activations were captured for Score-CAM.")

    acts = activations[0]
    acts_up = torch.nn.functional.interpolate(acts, size=(224, 224), mode="bilinear", align_corners=False)
    cam = torch.zeros(224, 224, device=input_tensor.device)

    with torch.no_grad():
        for channel_index in range(acts_up.shape[1]):
            act_map = acts_up[0, channel_index]
            act_min, act_max = act_map.min(), act_map.max()
            if act_max - act_min < 1e-8:
                continue
            act_norm = (act_map - act_min) / (act_max - act_min)
            masked = input_tensor * act_norm.unsqueeze(0).unsqueeze(0)
            score = model(masked).item()
            cam += score * act_map

    cam = torch.relu(cam)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.cpu().numpy()


def score_cam_vit(model, input_tensor):
    model.eval()
    activations = []

    def hook_fn(module, inputs, output):
        activations.append(output.detach())

    handle = model.blocks[-1].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(input_tensor)
    handle.remove()

    if not activations:
        raise RuntimeError("No ViT activations were captured for Score-CAM.")

    acts = activations[0]
    patch_tokens = acts[:, 1:, :]
    num_patches = int(np.sqrt(patch_tokens.shape[1]))
    feat_map = patch_tokens.reshape(
        1,
        num_patches,
        num_patches,
        patch_tokens.shape[-1],
    ).permute(0, 3, 1, 2)
    feat_up = torch.nn.functional.interpolate(feat_map, size=(224, 224), mode="bilinear", align_corners=False)
    cam = torch.zeros(224, 224, device=input_tensor.device)

    with torch.no_grad():
        for channel_index in range(feat_up.shape[1]):
            feature_map = feat_up[0, channel_index]
            feat_min, feat_max = feature_map.min(), feature_map.max()
            if feat_max - feat_min < 1e-8:
                continue
            feature_norm = (feature_map - feat_min) / (feat_max - feat_min)
            masked = input_tensor * feature_norm.unsqueeze(0).unsqueeze(0)
            score = model(masked).item()
            cam += score * feature_map

    cam = torch.relu(cam)
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.cpu().numpy()


def prepare_scorecam_rgb_image(raw, size=(224, 224)):
    image = raw.copy()
    if image.ndim == 3 and image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3, 4]:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    if image.max() > 1.5:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def normalize_scorecam(cam):
    cam = np.asarray(cam, dtype=np.float32)
    cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_LINEAR)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def plot_scorecam_3x3(results, representative_seed: int, args):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(8.4, 9.6), facecolor="white")
    column_titles = ["Input", "ResNet-18 Score-CAM", "ViT-Tiny Score-CAM"]
    for column_index, title in enumerate(column_titles):
        axes[0, column_index].set_title(title, fontsize=12, fontweight="bold", pad=8)

    last_cam_im = None
    for row_index, result in enumerate(results):
        case = result["case"]
        raw_img = prepare_scorecam_rgb_image(result["raw"])
        resnet_cam = normalize_scorecam(result["resnet18_cam"])
        vit_cam = normalize_scorecam(result["vit_tiny_cam"])
        category_short = str(case.get("category", "")).split("_")[0]
        case_number = str(case["label"]).replace("Case ", "")

        ax = axes[row_index, 0]
        ax.imshow(raw_img)
        ax.set_xlabel(
            f"Case {case_number} | {category_short} | Actual = {case['true_value']:.0f}\n"
            f"Extrapolated: {case['extrap_vars']}",
            fontsize=11,
            labelpad=6,
        )

        ax = axes[row_index, 1]
        ax.imshow(raw_img)
        last_cam_im = ax.imshow(resnet_cam, cmap=args.scorecam_cmap, alpha=args.scorecam_overlay_alpha, vmin=0, vmax=1)
        ax.set_xlabel(
            f"Pred = {case['resnet18_pred']:.0f}, Error = {case['resnet18_error']:+.0f}",
            fontsize=11,
            labelpad=6,
        )

        ax = axes[row_index, 2]
        ax.imshow(raw_img)
        last_cam_im = ax.imshow(vit_cam, cmap=args.scorecam_cmap, alpha=args.scorecam_overlay_alpha, vmin=0, vmax=1)
        ax.set_xlabel(
            f"Pred = {case['vit_tiny_pred']:.0f}, Error = {case['vit_tiny_error']:+.0f}",
            fontsize=11,
            labelpad=6,
        )

    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.subplots_adjust(left=0.03, right=0.98, top=0.96, bottom=0.10, wspace=0.05, hspace=0.20)
    cbar_ax = fig.add_axes([0.42, 0.045, 0.50, 0.015])
    cbar = fig.colorbar(last_cam_im, cax=cbar_ax, orientation="horizontal")
    cbar.ax.tick_params(labelsize=8, length=2)

    save_figure(fig, args.output_dir, "fig_4_3_scorecam_3cases", args.formats)


def run_scorecam_figure(args) -> None:
    load_scorecam_libraries()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_dfs = load_seed_predictions(args.evaluation_dir, args.train_seeds)
    seed_dfs = align_scorecam_seed_predictions(seed_dfs, args.train_seeds)
    overall_df = load_scorecam_overall_metrics(args, seed_dfs)

    representative_seed = args.representative_seed
    if representative_seed is None:
        representative_seed = select_representative_seed(overall_df, args.train_seeds, args.output_dir)

    avg_df = build_average_error_dataframe(seed_dfs, args.train_seeds, args.category_column)
    cases = select_scorecam_cases(avg_df, args.category_column)
    verification_df = verify_scorecam_cases(cases, seed_dfs, args.train_seeds)
    eval_df = load_eval_metadata(args)
    cases = enrich_scorecam_cases(cases, eval_df, args)

    image_dir = args.additional_eval_image_dir
    if image_dir is None:
        image_dir = args.additional_eval_data_dir / "layout_images"

    seed_model_dir = find_seed_model_dir(args.model_dir, representative_seed)
    resnet_model, resnet_mean, resnet_std = load_scorecam_model(
        "resnet18",
        find_model_path(seed_model_dir, "resnet18"),
        args,
        device,
    )
    vit_model, vit_mean, vit_std = load_scorecam_model(
        "vit_tiny",
        find_model_path(seed_model_dir, "vit_tiny"),
        args,
        device,
    )

    results = []
    for case in cases:
        image_path = resolve_scorecam_image_path(case["layout_image_path"], image_dir)
        input_tensor, raw_img = load_and_preprocess_scorecam_image(image_path, device)

        with torch.no_grad():
            resnet_pred = resnet_model(input_tensor).item() * resnet_std + resnet_mean
            vit_pred = vit_model(input_tensor).item() * vit_std + vit_mean

        case["resnet18_pred"] = float(resnet_pred)
        case["resnet18_error"] = float(resnet_pred - case["true_value"])
        case["vit_tiny_pred"] = float(vit_pred)
        case["vit_tiny_error"] = float(vit_pred - case["true_value"])

        results.append(
            {
                "case": case,
                "raw": raw_img,
                "resnet18_cam": score_cam_cnn(resnet_model, input_tensor),
                "vit_tiny_cam": score_cam_vit(vit_model, input_tensor),
            }
        )

    source_dir = args.output_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    verification_df.to_csv(source_dir / "scorecam_case_seed_verification.csv", index=False)
    pd.DataFrame(cases).to_csv(source_dir / "scorecam_representative_cases.csv", index=False)
    save_json(
        source_dir / "scorecam_config.json",
        {
            "representative_seed": int(representative_seed),
            "train_seeds": [int(seed) for seed in args.train_seeds],
            "model_dir": str(args.model_dir),
            "evaluation_dir": str(args.evaluation_dir),
            "image_dir": str(image_dir),
            "selection_rules": [
                "Case 1: minimum combined absolute error across model averages.",
                "Case 2: maximum ViT-Tiny minus ResNet-18 absolute error.",
                "Case 3: largest underprediction in the top true-risk quartile.",
            ],
        },
    )

    plot_scorecam_3x3(results, representative_seed, args)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate selected paper figures from experiment CSV outputs."
    )
    parser.add_argument("--training-csv", type=Path, default=Path("data/generated_dataset/trial_summary_all.csv"))
    parser.add_argument("--additional-eval-data-dir", type=Path, default=Path("data/additional_evaluation_dataset"))
    parser.add_argument("--additional-eval-csv", type=Path, default=None)
    parser.add_argument("--additional-eval-image-dir", type=Path, default=None)
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/additional_evaluation"))
    parser.add_argument("--category-metrics-csv", type=Path, default=None)
    parser.add_argument("--risk-level-metrics-csv", type=Path, default=None)
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/multi_seed_training"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_figures"))
    parser.add_argument("--figures", type=parse_figures, default=parse_figures("all"))
    parser.add_argument("--formats", type=parse_str_list, default=parse_str_list("png,pdf"))
    parser.add_argument("--train-seeds", type=parse_int_list, default=parse_int_list("42,43,44"))

    parser.add_argument("--category-column", default="category")
    parser.add_argument("--target-column", default="invaded_grid_count")
    parser.add_argument("--clean-column", default="clean_success")
    parser.add_argument("--fence-column", default="fence_x")
    parser.add_argument("--pillar-x-column", default="pillar_x")
    parser.add_argument("--pillar-y-column", default="pillar_y")

    parser.add_argument("--histogram-bins", type=int, default=40)
    parser.add_argument("--positive-quantile", type=float, default=0.75)
    parser.add_argument("--gain-points", type=int, default=50)
    parser.add_argument("--min-risk-count", type=int, default=300)
    parser.add_argument("--scorecam-overall-csv", type=Path, default=None)
    parser.add_argument("--representative-seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--scorecam-cmap", default="jet")
    parser.add_argument("--scorecam-overlay-alpha", type=float, default=0.48)
    parser.add_argument("--fence-range", type=parse_float_pair, default=parse_float_pair("0.35,0.50"))
    parser.add_argument("--pillar-x-range", type=parse_float_pair, default=parse_float_pair("-0.45,-0.25"))
    parser.add_argument("--pillar-y-range", type=parse_float_pair, default=parse_float_pair("-0.55,-0.35"))
    parser.add_argument("--pick-offset-range", type=parse_float_pair, default=parse_float_pair("-0.10,0.10"))
    parser.add_argument("--place-offset-range", type=parse_float_pair, default=parse_float_pair("-0.10,0.10"))
    parser.add_argument("--extrap-tolerance", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_plot_libraries()
    configure_style()

    valid_formats = {"png", "pdf", "svg"}
    invalid_formats = [fmt for fmt in args.formats if fmt.lower() not in valid_formats]
    if invalid_formats:
        raise ValueError(f"Unsupported output format(s): {invalid_formats}")
    args.formats = [fmt.lower() for fmt in args.formats]

    seed_dfs = None
    if "screening_gain" in args.figures or "target_distribution" in args.figures:
        try:
            seed_dfs = load_seed_predictions(args.evaluation_dir, args.train_seeds)
        except FileNotFoundError:
            if "screening_gain" in args.figures:
                raise
            seed_dfs = None

    if "target_distribution" in args.figures:
        plot_target_distribution(args, seed_dfs)
    if "category_summary" in args.figures:
        plot_category_summary(args)
    if "trend_summary" in args.figures:
        plot_trend_summary(args)
    if "screening_gain" in args.figures:
        plot_screening_gain(args, seed_dfs)
    if "risk_level_summary" in args.figures:
        plot_risk_level_summary(args)
    if "fence_pillar" in args.figures:
        plot_fence_pillar_combined(args)
    if "scorecam" in args.figures:
        run_scorecam_figure(args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
