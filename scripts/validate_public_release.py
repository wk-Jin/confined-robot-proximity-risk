"""Validate the public data and model layout after extracting release assets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_MODELS = ["resnet18_layout_risk.pt", "vit_tiny_layout_risk.pt"]
DEFAULT_CATEGORIES = ["C0", "C1", "C2", "C3", "C4"]


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one seed.")
    return values


def resolve_image_path(raw_value, image_dir: Path) -> Path:
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text.lower() == "nan":
        raise FileNotFoundError("Empty image path in CSV.")

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
    raise FileNotFoundError(f"Image file not found. Searched: {searched}")


def check_required_columns(dataframe, required_columns: list[str], csv_path: Path) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")


def check_image_paths(dataframe, image_dir: Path, image_column: str, max_checks: int | None, label: str) -> None:
    if image_column not in dataframe.columns:
        raise ValueError(f"{label} CSV is missing image column: {image_column}")

    paths = dataframe[image_column]
    if max_checks is not None:
        paths = paths.head(max_checks)

    for raw_value in paths:
        resolve_image_path(raw_value, image_dir)

    checked = len(paths)
    suffix = "" if max_checks is None else f" of {len(dataframe)}"
    print(f"[OK] {label}: verified {checked}{suffix} image paths")


def validate_generated_dataset(args) -> None:
    csv_path = args.generated_data_dir / "trial_summary_all.csv"
    image_dir = args.generated_data_dir / "layout_images"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing generated dataset CSV: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing generated dataset image directory: {image_dir}")

    dataframe = pd.read_csv(csv_path)
    check_required_columns(
        dataframe,
        ["layout_image_path", "invaded_grid_count", "clean_success"],
        csv_path,
    )
    check_image_paths(dataframe, image_dir, "layout_image_path", args.max_image_checks, "generated dataset")
    print(f"[OK] generated dataset rows: {len(dataframe)}")


def normalize_category(value: str) -> str:
    text = str(value)
    if "_" in text:
        return text.split("_", 1)[0]
    return text


def validate_additional_evaluation_dataset(args) -> None:
    csv_path = args.additional_eval_data_dir / "additional_evaluation_dataset.csv"
    image_dir = args.additional_eval_data_dir / "layout_images"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing additional evaluation CSV: {csv_path}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Missing additional evaluation image directory: {image_dir}")

    dataframe = pd.read_csv(csv_path)
    check_required_columns(
        dataframe,
        ["category", "layout_image_path", "invaded_grid_count"],
        csv_path,
    )

    if args.expected_additional_rows > 0 and len(dataframe) != args.expected_additional_rows:
        raise ValueError(
            f"Expected {args.expected_additional_rows} additional evaluation rows, found {len(dataframe)}."
        )

    category_counts = dataframe["category"].map(normalize_category).value_counts().sort_index()
    for category in DEFAULT_CATEGORIES:
        count = int(category_counts.get(category, 0))
        if args.expected_per_category > 0 and count != args.expected_per_category:
            raise ValueError(
                f"Expected {args.expected_per_category} rows for {category}, found {count}."
            )

    check_image_paths(
        dataframe,
        image_dir,
        "layout_image_path",
        args.max_image_checks,
        "additional evaluation dataset",
    )
    print("[OK] additional evaluation category counts:")
    print(category_counts.to_string())


def validate_models(args) -> None:
    for seed in args.train_seeds:
        seed_dir = args.model_dir / f"seed_{seed}"
        if not seed_dir.exists():
            raise FileNotFoundError(f"Missing seed model directory: {seed_dir}")
        for file_name in DEFAULT_MODELS:
            path = seed_dir / file_name
            if not path.exists():
                raise FileNotFoundError(f"Missing model checkpoint: {path}")
            print(f"[OK] model checkpoint: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate extracted release assets.")
    parser.add_argument("--generated-data-dir", type=Path, default=Path("data/generated_dataset"))
    parser.add_argument("--additional-eval-data-dir", type=Path, default=Path("data/additional_evaluation_dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/multi_seed_training"))
    parser.add_argument("--train-seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--max-image-checks", type=int, default=None)
    parser.add_argument("--expected-additional-rows", type=int, default=1000)
    parser.add_argument("--expected-per-category", type=int, default=200)
    parser.add_argument("--skip-generated-dataset", action="store_true")
    parser.add_argument("--skip-additional-evaluation", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.skip_generated_dataset:
        validate_generated_dataset(args)
    if not args.skip_additional_evaluation:
        validate_additional_evaluation_dataset(args)
    if not args.skip_models:
        validate_models(args)

    print("[OK] public release validation complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
