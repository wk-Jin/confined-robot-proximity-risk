"""Build the three GitHub Release archives for the public repository."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


DEFAULT_SEEDS = [42, 43, 44]
MODEL_FILES = ["resnet18_layout_risk.pt", "vit_tiny_layout_risk.pt"]


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Provide at least one seed.")
    return values


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def add_directory_to_zip(zip_path: Path, source_dir: Path, archive_root: str | None) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source directory: {source_dir}")

    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in iter_files(source_dir):
            relative = path.relative_to(source_dir)
            arcname = Path(archive_root) / relative if archive_root else relative
            zf.write(path, arcname.as_posix())
    print(f"Created: {zip_path}")


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")


def validate_generated_dataset(path: Path) -> None:
    require_file(path / "trial_summary_all.csv")
    if not (path / "layout_images").exists():
        raise FileNotFoundError(f"Missing layout image directory: {path / 'layout_images'}")


def validate_additional_evaluation_dataset(path: Path) -> None:
    require_file(path / "additional_evaluation_dataset.csv")
    if not (path / "layout_images").exists():
        raise FileNotFoundError(f"Missing layout image directory: {path / 'layout_images'}")


def validate_models(path: Path, seeds: list[int]) -> None:
    for seed in seeds:
        seed_dir = path / f"seed_{seed}"
        if not seed_dir.exists():
            raise FileNotFoundError(f"Missing seed model directory: {seed_dir}")
        for file_name in MODEL_FILES:
            require_file(seed_dir / file_name)


def build_model_zip(zip_path: Path, model_dir: Path, seeds: list[int]) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for seed in seeds:
            seed_dir = model_dir / f"seed_{seed}"
            for path in iter_files(seed_dir):
                arcname = path.relative_to(model_dir)
                zf.write(path, arcname.as_posix())
    print(f"Created: {zip_path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Create GitHub Release zip archives.")
    parser.add_argument("--generated-data-dir", type=Path, default=Path("data/generated_dataset"))
    parser.add_argument("--additional-eval-data-dir", type=Path, default=Path("data/additional_evaluation_dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/multi_seed_training"))
    parser.add_argument("--release-dir", type=Path, default=Path("release_assets"))
    parser.add_argument("--train-seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--skip-generated-dataset", action="store_true")
    parser.add_argument("--skip-additional-evaluation", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.skip_generated_dataset:
        validate_generated_dataset(args.generated_data_dir)
        add_directory_to_zip(
            args.release_dir / "generated_dataset.zip",
            args.generated_data_dir,
            archive_root="generated_dataset",
        )

    if not args.skip_additional_evaluation:
        validate_additional_evaluation_dataset(args.additional_eval_data_dir)
        add_directory_to_zip(
            args.release_dir / "additional_evaluation_dataset.zip",
            args.additional_eval_data_dir,
            archive_root="additional_evaluation_dataset",
        )

    if not args.skip_models:
        validate_models(args.model_dir, args.train_seeds)
        build_model_zip(args.release_dir / "trained_models.zip", args.model_dir, args.train_seeds)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
