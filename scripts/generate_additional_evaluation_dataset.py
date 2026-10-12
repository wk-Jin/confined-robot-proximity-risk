"""
Generate the additional interpolation/extrapolation evaluation dataset.

Sequential sampling procedure for the C0-C4 additional evaluation:

1. Enumerate parameter combinations outside the model-development grid.
2. For each category, sort all combinations by combination_id and randomly
   order them once with numpy.random.RandomState(seed).
3. Simulate the combinations of each category in that fixed order until
   --target-per-category clean-success samples have been obtained
   (categories are processed from C4 to C0).
4. The final dataset contains, for each category, the first
   --target-per-category clean-success samples in the fixed order.
5. Layout images are generated only for the final dataset rows.

The final dataset is written only when every category has exactly
--target-per-category clean-success samples. If a category runs out of
combinations first, the script stops with an error and writes no final
dataset. Resume is keyed by combination_id, so an interrupted run continues
from where it stopped.

Outputs:
    - all_combinations.csv: all enumerated candidate conditions and labels.
    - simulation_order.csv: fixed per-category simulation order.
    - candidate_simulation_results.csv: simulation summaries of simulated rows.
    - generation_summary.csv: simulated/clean/error counts per category.
    - additional_evaluation_dataset.csv: selected clean-success evaluation set.
    - layout_images/*.npy: model-input layout images of the final dataset only.

Example:
    python generate_additional_evaluation_dataset.py --yes
    python generate_additional_evaluation_dataset.py --stats
"""

import argparse
import csv
import importlib
import inspect
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

for import_path in (SCRIPT_DIR, PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)


RANDOM_SEED = 42

CATEGORIES = ["C0", "C1", "C2", "C3", "C4"]
# Simulation order of categories: the lowest clean-success rate first.
RUN_CATEGORY_ORDER = ["C4", "C3", "C2", "C1", "C0"]
DEFAULT_TARGET_PER_CATEGORY = 200


# Model-development ranges from Table 2.
FENCE_TRAIN = [0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500]
PILLAR_X_TRAIN = [-0.45, -0.35, -0.25]
PILLAR_Y_TRAIN = [-0.55, -0.45, -0.35]
PICK_TRAIN = [-0.10, -0.05, 0.00, 0.05, 0.10]
PLACE_TRAIN = [-0.10, -0.05, 0.00, 0.05, 0.10]

FENCE_MIN, FENCE_MAX = min(FENCE_TRAIN), max(FENCE_TRAIN)
PILLAR_X_MIN, PILLAR_X_MAX = min(PILLAR_X_TRAIN), max(PILLAR_X_TRAIN)
PILLAR_Y_MIN, PILLAR_Y_MAX = min(PILLAR_Y_TRAIN), max(PILLAR_Y_TRAIN)
PICK_MIN, PICK_MAX = min(PICK_TRAIN), max(PICK_TRAIN)
PLACE_MIN, PLACE_MAX = min(PLACE_TRAIN), max(PLACE_TRAIN)


# Additional evaluation values from Appendix A.2.
FENCE_INTERP = [0.3625, 0.3875, 0.4125, 0.4375, 0.4625, 0.4875]
FENCE_EXTRAP = [0.300, 0.325, 0.525, 0.550]
FENCE_ALL = sorted(set(FENCE_INTERP + FENCE_EXTRAP))

PILLAR_INTERP_COMBOS = [
    (-0.40, -0.50),
    (-0.40, -0.40),
    (-0.30, -0.50),
    (-0.30, -0.40),
]

PILLAR_1AXIS_EXTRAP = [
    (-0.55, -0.55),
    (-0.55, -0.45),
    (-0.55, -0.35),
    (-0.20, -0.55),
    (-0.20, -0.45),
    (-0.20, -0.35),
    (-0.45, -0.60),
    (-0.35, -0.60),
    (-0.25, -0.60),
    (-0.45, -0.30),
    (-0.35, -0.30),
    (-0.25, -0.30),
]

PILLAR_2AXIS_EXTRAP = [
    (-0.55, -0.60),
    (-0.55, -0.30),
    (-0.20, -0.60),
    (-0.20, -0.30),
]

PILLAR_ALL_COMBOS = (
    PILLAR_INTERP_COMBOS + PILLAR_1AXIS_EXTRAP + PILLAR_2AXIS_EXTRAP
)

OFFSET_INTERP_VALUES = [-0.075, -0.025, 0.025, 0.075]
OFFSET_EXTRAP_VALUES = [-0.125, 0.000, 0.125]

PICK_INTERP_COMBOS = [
    (x, y) for x in OFFSET_INTERP_VALUES for y in OFFSET_INTERP_VALUES
]
PLACE_INTERP_COMBOS = [
    (x, y) for x in OFFSET_INTERP_VALUES for y in OFFSET_INTERP_VALUES
]

PICK_EXTRAP_COMBOS = [
    (x, y)
    for x in OFFSET_EXTRAP_VALUES
    for y in OFFSET_EXTRAP_VALUES
    if (x, y) != (0.0, 0.0)
]
PLACE_EXTRAP_COMBOS = PICK_EXTRAP_COMBOS.copy()

PICK_ALL_COMBOS = PICK_INTERP_COMBOS + PICK_EXTRAP_COMBOS
PLACE_ALL_COMBOS = PLACE_INTERP_COMBOS + PLACE_EXTRAP_COMBOS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the C0-C4 additional evaluation dataset."
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "additional_evaluation_dataset"),
        help="Directory for generated CSV files and layout images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used for the fixed per-category simulation order.",
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=DEFAULT_TARGET_PER_CATEGORY,
        help="Number of clean-success samples required for each C0-C4 category.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Start simulation without an interactive confirmation prompt.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run PyBullet with the GUI instead of DIRECT mode.",
    )
    parser.add_argument(
        "--save-layout-images",
        action="store_true",
        help=(
            "Kept for compatibility; no effect. 128x128 layout images are always "
            "written for the final dataset only."
        ),
    )
    parser.add_argument(
        "--save-step-data",
        action="store_true",
        help="Also save per-step records. This can produce a very large CSV.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Save simulation progress after every N newly completed candidates.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not reuse candidate_simulation_results.csv if it already exists.",
    )
    parser.add_argument(
        "--combinations-only",
        action="store_true",
        help="Only write all_combinations.csv and simulation_order.csv.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics for existing generated files.",
    )
    return parser.parse_args()


def output_paths(output_dir):
    root = Path(output_dir).expanduser().resolve()
    return {
        "root": root,
        "layout_images": root / "layout_images",
        "all_combinations": root / "all_combinations.csv",
        "simulation_order": root / "simulation_order.csv",
        "simulation_results": root / "candidate_simulation_results.csv",
        "generation_summary": root / "generation_summary.csv",
        "evaluation_dataset": root / "additional_evaluation_dataset.csv",
        "step_data": root / "step_data_all.csv",
        "progress_log": root / "progress.log",
    }


def ensure_output_dirs(paths):
    paths["root"].mkdir(parents=True, exist_ok=True)
    paths["layout_images"].mkdir(parents=True, exist_ok=True)


def log_progress(paths, message):
    print(message)
    with open(paths["progress_log"], "a", encoding="utf-8") as handle:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        handle.write(f"[{timestamp}] {message}\n")


def is_outside(value, lower, upper, tol=1e-4):
    return value < lower - tol or value > upper + tol


def is_fence_extrapolated(fence_x):
    return is_outside(fence_x, FENCE_MIN, FENCE_MAX)


def is_pillar_extrapolated(pillar_x, pillar_y):
    return (
        is_outside(pillar_x, PILLAR_X_MIN, PILLAR_X_MAX)
        or is_outside(pillar_y, PILLAR_Y_MIN, PILLAR_Y_MAX)
    )


def is_pick_extrapolated(pick_x, pick_y):
    return is_outside(pick_x, PICK_MIN, PICK_MAX) or is_outside(
        pick_y, PICK_MIN, PICK_MAX
    )


def is_place_extrapolated(place_x, place_y):
    return is_outside(place_x, PLACE_MIN, PLACE_MAX) or is_outside(
        place_y, PLACE_MIN, PLACE_MAX
    )


def categorize_combination(
    fence_x,
    pillar_x,
    pillar_y,
    pick_x_offset,
    pick_y_offset,
    place_x_offset,
    place_y_offset,
):
    flags = {
        "fence_extrap": is_fence_extrapolated(fence_x),
        "pillar_extrap": is_pillar_extrapolated(pillar_x, pillar_y),
        "pick_extrap": is_pick_extrapolated(pick_x_offset, pick_y_offset),
        "place_extrap": is_place_extrapolated(place_x_offset, place_y_offset),
    }
    n_extrap = int(sum(flags.values()))
    category = f"C{n_extrap}"
    return n_extrap, category, flags


def generate_all_combinations():
    rows = []
    for fence_x in FENCE_ALL:
        for pillar_x, pillar_y in PILLAR_ALL_COMBOS:
            for pick_x, pick_y in PICK_ALL_COMBOS:
                for place_x, place_y in PLACE_ALL_COMBOS:
                    n_extrap, category, flags = categorize_combination(
                        fence_x,
                        pillar_x,
                        pillar_y,
                        pick_x,
                        pick_y,
                        place_x,
                        place_y,
                    )
                    rows.append(
                        {
                            "fence_x": fence_x,
                            "pillar_x": pillar_x,
                            "pillar_y": pillar_y,
                            "pick_x_offset": pick_x,
                            "pick_y_offset": pick_y,
                            "place_x_offset": place_x,
                            "place_y_offset": place_y,
                            "category": category,
                            "n_extrapolated_groups": n_extrap,
                            **flags,
                        }
                    )

    df = pd.DataFrame(rows)
    df.insert(0, "combination_id", range(len(df)))
    return df


def build_simulation_order(df, seed):
    """Fixed per-category simulation order.

    Each category is sorted by combination_id and permuted once with
    numpy.random.RandomState(seed), whose stream is stable across numpy
    versions. candidate_id is set equal to combination_id so that row
    identity does not depend on the order or on the number simulated.
    """
    ordered = []
    for category in RUN_CATEGORY_ORDER:
        category_df = (
            df[df["category"] == category]
            .sort_values("combination_id")
            .reset_index(drop=True)
        )
        if category_df.empty:
            continue
        permutation = np.random.RandomState(seed).permutation(len(category_df))
        category_df = category_df.iloc[permutation].reset_index(drop=True)
        category_df.insert(0, "order_in_category", range(len(category_df)))
        ordered.append(category_df)

    if not ordered:
        return pd.DataFrame(columns=["candidate_id", "order_in_category", *df.columns])

    result = pd.concat(ordered, ignore_index=True)
    result.insert(0, "candidate_id", result["combination_id"].astype(int))
    return result


def load_simulation_module():
    module_name = os.environ.get("FRANKA_SIMULATION_MODULE", "franka_simulation")
    return importlib.import_module(module_name)


def run_simulation_wrapper(simulation_module, row, headless):
    layout_params = {
        "layout_id": int(row["candidate_id"]),
        "fence_x": float(row["fence_x"]),
        "pillar_x": float(row["pillar_x"]),
        "pillar_y": float(row["pillar_y"]),
    }
    trial_params = {
        "trial_id": int(row["candidate_id"]),
        "pick_x_offset": float(row["pick_x_offset"]),
        "pick_y_offset": float(row["pick_y_offset"]),
        "place_x_offset": float(row["place_x_offset"]),
        "place_y_offset": float(row["place_y_offset"]),
    }

    kwargs = {
        "layout_params": layout_params,
        "trial_params": trial_params,
        "headless": headless,
        "verbose": False,
        "save_step_data": False,
        "save_layout_image": False,
    }
    if "save_xy_figure" in inspect.signature(simulation_module.run_simulation).parameters:
        kwargs["save_xy_figure"] = False

    return simulation_module.run_simulation(**kwargs)


def write_step_data(step_logger, row, path, append=True):
    if step_logger is None or not getattr(step_logger, "records", None):
        return

    records = []
    for record in step_logger.records:
        updated = dict(record)
        updated["candidate_id"] = int(row["candidate_id"])
        updated["category"] = row["category"]
        updated["n_extrapolated_groups"] = int(row["n_extrapolated_groups"])
        records.append(updated)

    fieldnames = list(records[0].keys())
    for column in ["n_extrapolated_groups", "category", "candidate_id"]:
        if column in fieldnames:
            fieldnames.remove(column)
            fieldnames.insert(0, column)

    file_exists = Path(path).exists()
    mode = "a" if append and file_exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(records)


def save_layout_image(simulation_module, row, paths):
    image_name = f"candidate_{int(row['candidate_id']):04d}.npy"
    image_path = paths["layout_images"] / image_name
    layout_params = {
        "fence_x": float(row["fence_x"]),
        "pillar_x": float(row["pillar_x"]),
        "pillar_y": float(row["pillar_y"]),
    }
    trial_params = {
        "pick_x_offset": float(row["pick_x_offset"]),
        "pick_y_offset": float(row["pick_y_offset"]),
        "place_x_offset": float(row["place_x_offset"]),
        "place_y_offset": float(row["place_y_offset"]),
    }
    generator = simulation_module.LayoutImageGenerator()
    generator.generate(layout_params, trial_params)
    generator.save(str(image_path))
    return (Path("layout_images") / image_name).as_posix()


def clean_result_row(row, sim_result, sim_time_s, layout_image_path=None):
    result = {
        "candidate_id": int(row["candidate_id"]),
        "combination_id": int(row["combination_id"]),
        "category": row["category"],
        "n_extrapolated_groups": int(row["n_extrapolated_groups"]),
        "fence_extrap": bool(row["fence_extrap"]),
        "pillar_extrap": bool(row["pillar_extrap"]),
        "pick_extrap": bool(row["pick_extrap"]),
        "place_extrap": bool(row["place_extrap"]),
        "sim_time_s": float(sim_time_s),
    }

    for key, value in sim_result.items():
        if key.startswith("_"):
            continue
        result[key] = value

    if layout_image_path is not None:
        result["layout_image_path"] = layout_image_path

    return result


PARAMETER_COLUMNS = [
    "fence_x",
    "pillar_x",
    "pillar_y",
    "pick_x_offset",
    "pick_y_offset",
    "place_x_offset",
    "place_y_offset",
]


def load_existing_results(order_df, paths, args):
    """Load previous results keyed by combination_id and check consistency."""
    results_path = paths["simulation_results"]
    if args.no_resume or not results_path.exists():
        return {}

    existing_df = pd.read_csv(results_path)
    if existing_df.empty:
        return {}
    if "combination_id" not in existing_df.columns:
        raise SystemExit(
            f"{results_path} has no combination_id column and cannot be resumed. "
            "Use a new --output-dir or --no-resume."
        )

    existing_df = existing_df.drop_duplicates("combination_id", keep="first")
    reference = order_df.set_index("combination_id")
    for _, row in existing_df.iterrows():
        combination_id = int(row["combination_id"])
        if combination_id not in reference.index:
            raise SystemExit(
                f"Resume check failed: combination_id {combination_id} in {results_path} "
                "is not in the current combination list."
            )
        expected = reference.loc[combination_id]
        if str(row["category"]) != str(expected["category"]):
            raise SystemExit(
                f"Resume check failed: category mismatch for combination_id {combination_id}."
            )
        for column in PARAMETER_COLUMNS:
            if column in row.index and pd.notna(row[column]):
                if abs(float(row[column]) - float(expected[column])) > 1e-6:
                    raise SystemExit(
                        f"Resume check failed: {column} mismatch for "
                        f"combination_id {combination_id}."
                    )

    log_progress(paths, f"Resume: {len(existing_df)} simulated combinations loaded")
    return {
        int(record["combination_id"]): record
        for record in existing_df.to_dict("records")
    }


def simulate_one(simulation_module, row, paths, args):
    sim_started = time.time()
    try:
        sim_result = run_simulation_wrapper(
            simulation_module=simulation_module,
            row=row,
            headless=not args.gui,
        )
        # Layout images are not saved here. They are generated only for the
        # final dataset after selection (see write_final_layout_images).
        layout_image_path = None
        if args.save_step_data:
            write_step_data(sim_result.get("_step_logger"), row, paths["step_data"])
    except Exception as exc:
        sim_result = {
            "clean_success": False,
            "simulation_error": str(exc),
        }
        layout_image_path = None

    sim_time_s = time.time() - sim_started
    return clean_result_row(row, sim_result, sim_time_s, layout_image_path)


def run_until_target(order_df, paths, args):
    """Simulate each category in its fixed order until the target is reached.

    Every position before the stopping point has a result, so the first
    target clean-success rows by order_in_category form the final dataset.
    If a category runs out of combinations first, the results are saved and
    the script stops with an error.
    """
    simulation_module = None
    results_path = paths["simulation_results"]
    results_by_id = load_existing_results(order_df, paths, args)
    target = args.target_per_category

    started = time.time()
    newly_completed = 0
    summary_rows = []

    def save_results():
        pd.DataFrame(list(results_by_id.values())).to_csv(results_path, index=False)

    for category in RUN_CATEGORY_ORDER:
        category_df = order_df[order_df["category"] == category]
        n_clean = 0
        n_error = 0
        n_processed = 0
        stop_position = None

        for _, row in category_df.iterrows():
            combination_id = int(row["combination_id"])
            record = results_by_id.get(combination_id)
            if record is None:
                if simulation_module is None:
                    simulation_module = load_simulation_module()
                record = simulate_one(simulation_module, row, paths, args)
                results_by_id[combination_id] = record
                newly_completed += 1

                if args.save_every > 0 and newly_completed % args.save_every == 0:
                    save_results()
                    elapsed = time.time() - started
                    avg = elapsed / max(newly_completed, 1)
                    log_progress(
                        paths,
                        f"Progress {category}: clean {n_clean}/{target}, "
                        f"processed {n_processed + 1}, avg={avg:.1f}s/sim",
                    )

            n_processed += 1
            if is_clean_success(record.get("clean_success")):
                n_clean += 1
            error_value = record.get("simulation_error")
            if error_value is not None and pd.notna(error_value) and str(error_value):
                n_error += 1

            if n_clean >= target:
                stop_position = int(row["order_in_category"])
                break

        save_results()
        summary_rows.append(
            {
                "category": category,
                "total_combinations": int(len(category_df)),
                "n_simulated": int(n_processed),
                "n_clean": int(n_clean),
                "n_non_clean": int(n_processed - n_clean),
                "n_simulation_error": int(n_error),
                "clean_rate": float(n_clean / n_processed) if n_processed else float("nan"),
                "stop_order_in_category": stop_position,
                "seed": int(args.seed),
            }
        )
        pd.DataFrame(summary_rows).to_csv(paths["generation_summary"], index=False)
        log_progress(
            paths,
            f"{category}: clean {n_clean}/{target} after {n_processed} simulated "
            f"of {len(category_df)} combinations",
        )

        if n_clean < target:
            raise SystemExit(
                f"{category} ran out of combinations: only {n_clean} clean-success "
                f"samples among all {len(category_df)} combinations "
                f"(target {target}). No final dataset was written."
            )

    result_df = pd.DataFrame(list(results_by_id.values()))
    return result_df, pd.DataFrame(summary_rows)


def is_clean_success(value):
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "1.0", "true", "yes"}
    if isinstance(value, float) and np.isnan(value):
        return False
    return bool(value)


def select_clean_evaluation_dataset(result_df, order_df, target_per_category):
    """First target clean-success rows of each category in the fixed order."""
    if "clean_success" not in result_df.columns:
        raise ValueError("Simulation results do not contain a clean_success column.")

    reference = order_df[["combination_id", "order_in_category"] + PARAMETER_COLUMNS].rename(
        columns={column: f"{column}__order" for column in PARAMETER_COLUMNS}
    )
    merged = result_df.drop(columns=["order_in_category"], errors="ignore").merge(
        reference, on="combination_id", how="left", validate="one_to_one"
    )
    clean_df = merged[merged["clean_success"].map(is_clean_success)].copy()

    selected = []
    for category in CATEGORIES:
        category_df = clean_df[clean_df["category"] == category].sort_values(
            "order_in_category"
        )
        if len(category_df) < target_per_category:
            raise SystemExit(
                f"{category}: only {len(category_df)} clean-success samples "
                f"(target {target_per_category}). No final dataset was written."
            )
        selected.append(category_df.head(target_per_category))

    final_df = pd.concat(selected, ignore_index=True)
    final_df.insert(0, "evaluation_id", range(len(final_df)))
    validate_evaluation_dataset(final_df, target_per_category)
    return final_df.drop(columns=[f"{column}__order" for column in PARAMETER_COLUMNS])


def write_final_layout_images(final_df, paths):
    """Generate layout images for the final dataset rows only.

    Images depend only on the layout and trial parameters, so they are
    generated after selection with the same LayoutImageGenerator used for
    the model-development dataset.
    """
    simulation_module = load_simulation_module()
    final_df = final_df.copy()
    image_paths = []
    for _, row in final_df.iterrows():
        image_paths.append(save_layout_image(simulation_module, row, paths))
    final_df["layout_image_path"] = image_paths

    expected_names = {Path(path).name for path in image_paths}
    missing = [name for name in expected_names if not (paths["layout_images"] / name).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} final layout images were not written.")

    extra = [
        path.name
        for path in paths["layout_images"].glob("*.npy")
        if path.name not in expected_names
    ]
    if extra:
        print(
            f"Warning: {paths['layout_images']} contains {len(extra)} .npy files "
            "that are not part of the final dataset."
        )
    return final_df


def validate_evaluation_dataset(final_df, target_per_category):
    """Checks run before the final dataset is written."""
    problems = []

    counts = final_df["category"].value_counts()
    for category in CATEGORIES:
        if int(counts.get(category, 0)) != target_per_category:
            problems.append(
                f"{category} has {int(counts.get(category, 0))} rows, "
                f"expected {target_per_category}"
            )

    if not final_df["clean_success"].map(is_clean_success).all():
        problems.append("non-clean-success rows present")

    if final_df["combination_id"].duplicated().any():
        problems.append("duplicated combination_id")

    for _, row in final_df.iterrows():
        combination_id = int(row["combination_id"])
        reference_values = [float(row[f"{column}__order"]) for column in PARAMETER_COLUMNS]
        _, category, _ = categorize_combination(*reference_values)
        if category != row["category"]:
            problems.append(
                f"combination_id {combination_id}: stored category "
                f"{row['category']} but parameters give {category}"
            )
            break
        for column, expected in zip(PARAMETER_COLUMNS, reference_values):
            if column in row.index and pd.notna(row[column]):
                if abs(float(row[column]) - expected) > 1e-6:
                    problems.append(
                        f"combination_id {combination_id}: simulated {column} "
                        f"{row[column]} differs from the combination list {expected}"
                    )
                    break

    if problems:
        raise SystemExit("Final dataset validation failed: " + "; ".join(problems))


def print_counts(label, df):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    if df.empty or "category" not in df.columns:
        print("No rows available.")
        return
    print(df["category"].value_counts().sort_index().to_string())


def print_existing_stats(paths):
    for key in ["all_combinations", "simulation_order", "simulation_results", "evaluation_dataset"]:
        path = paths[key]
        if not path.exists():
            print(f"{path.name}: not found")
            continue
        df = pd.read_csv(path)
        print_counts(path.name, df)
        if "clean_success" in df.columns:
            clean_count = df["clean_success"].map(is_clean_success).sum()
            print(f"clean_success: {clean_count}/{len(df)}")

    summary_path = paths["generation_summary"]
    if summary_path.exists():
        print("\n" + "=" * 70)
        print(summary_path.name)
        print("=" * 70)
        print(pd.read_csv(summary_path).to_string(index=False))


def confirm_or_exit(args, candidate_count):
    if args.yes or args.combinations_only or args.stats:
        return

    print(
        "This will run PyBullet simulations in the fixed per-category order until "
        f"{args.target_per_category} clean-success samples are obtained for each "
        f"category (at most {candidate_count} combinations in total)."
    )
    answer = input("Continue? (y/n): ").strip().lower()
    if answer != "y":
        raise SystemExit("Cancelled.")


def main():
    args = parse_args()
    paths = output_paths(args.output_dir)
    ensure_output_dirs(paths)
    np.random.seed(args.seed)

    if args.stats:
        print_existing_stats(paths)
        return

    all_df = generate_all_combinations()
    all_df.to_csv(paths["all_combinations"], index=False)
    print_counts("All combinations", all_df)

    order_df = build_simulation_order(all_df, seed=args.seed)
    order_df.to_csv(paths["simulation_order"], index=False)
    print_counts("Simulation order", order_df)

    if args.combinations_only:
        print(f"\nWrote combinations to: {paths['root']}")
        return

    if paths["evaluation_dataset"].exists():
        paths["evaluation_dataset"].unlink()

    confirm_or_exit(args, len(order_df))
    result_df, summary_df = run_until_target(order_df, paths, args)
    print_counts("Candidate simulation results", result_df)

    evaluation_df = select_clean_evaluation_dataset(
        result_df,
        order_df,
        target_per_category=args.target_per_category,
    )
    evaluation_df = write_final_layout_images(evaluation_df, paths)
    evaluation_df.to_csv(paths["evaluation_dataset"], index=False)
    print_counts("Selected additional evaluation dataset", evaluation_df)

    print("\n" + "=" * 70)
    print("Generation summary")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print(f"\nOutput directory: {paths['root']}")


if __name__ == "__main__":
    main()
