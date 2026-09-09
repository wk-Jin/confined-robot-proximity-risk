"""
Generate the additional interpolation/extrapolation evaluation dataset.

The script follows the two-stage sampling procedure described for the C0-C4
additional evaluation in the paper:

1. Enumerate parameter combinations outside the model-development grid.
2. Sample a fixed candidate pool from each C0-C4 category.
3. Run the PyBullet simulation for each candidate.
4. Select clean-success samples from each category for final evaluation.

Outputs:
    - all_combinations.csv: all enumerated candidate conditions and labels.
    - sampled_candidate_pool.csv: sampled candidate pool before simulation.
    - candidate_simulation_results.csv: simulation summaries for the pool.
    - additional_evaluation_dataset.csv: selected clean-success evaluation set.
    - layout_images/*.npy: optional model-input layout images.

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
DEFAULT_POOL_PER_CATEGORY = 400
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
        help="Random seed used for candidate-pool and final clean-sample selection.",
    )
    parser.add_argument(
        "--pool-per-category",
        type=int,
        default=DEFAULT_POOL_PER_CATEGORY,
        help="Number of candidates sampled before simulation for each C0-C4 category.",
    )
    parser.add_argument(
        "--target-per-category",
        type=int,
        default=DEFAULT_TARGET_PER_CATEGORY,
        help="Number of clean-success samples selected for each C0-C4 category.",
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
        help="Save 128x128 model-input layout images for simulated candidates.",
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
        help="Only write all_combinations.csv and sampled_candidate_pool.csv.",
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
        "candidate_pool": root / "sampled_candidate_pool.csv",
        "simulation_results": root / "candidate_simulation_results.csv",
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


def sample_by_category(df, per_category, seed):
    sampled = []
    for category in CATEGORIES:
        category_df = df[df["category"] == category]
        n_take = min(per_category, len(category_df))
        if n_take == 0:
            continue
        sampled.append(category_df.sample(n=n_take, random_state=seed))

    if not sampled:
        return pd.DataFrame(columns=df.columns)

    result = pd.concat(sampled, ignore_index=True)
    result = result.sample(frac=1, random_state=seed).reset_index(drop=True)
    result.insert(0, "candidate_id", range(len(result)))
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


def run_candidate_simulations(candidate_df, paths, args):
    simulation_module = load_simulation_module()
    results_path = paths["simulation_results"]

    existing_results = []
    completed = set()
    if not args.no_resume and results_path.exists():
        existing_df = pd.read_csv(results_path)
        existing_results = existing_df.to_dict("records")
        completed = set(existing_df["candidate_id"].astype(int).tolist())
        log_progress(paths, f"Resume: {len(completed)} candidates already completed")

    results = list(existing_results)
    total = len(candidate_df)
    started = time.time()
    newly_completed = 0

    for _, row in candidate_df.iterrows():
        candidate_id = int(row["candidate_id"])
        if candidate_id in completed:
            continue

        sim_started = time.time()
        try:
            sim_result = run_simulation_wrapper(
                simulation_module=simulation_module,
                row=row,
                headless=not args.gui,
            )
            layout_image_path = None
            if args.save_layout_images:
                layout_image_path = save_layout_image(simulation_module, row, paths)
            if args.save_step_data:
                write_step_data(sim_result.get("_step_logger"), row, paths["step_data"])
        except Exception as exc:
            sim_result = {
                "clean_success": False,
                "simulation_error": str(exc),
            }
            layout_image_path = None

        sim_time_s = time.time() - sim_started
        results.append(clean_result_row(row, sim_result, sim_time_s, layout_image_path))
        newly_completed += 1

        if args.save_every > 0 and newly_completed % args.save_every == 0:
            pd.DataFrame(results).to_csv(results_path, index=False)
            elapsed = time.time() - started
            remaining = max(total - len(results), 0)
            avg = elapsed / max(newly_completed, 1)
            log_progress(
                paths,
                f"Progress: {len(results)}/{total}, "
                f"avg={avg:.1f}s, remaining~{remaining * avg / 60:.1f}min",
            )

    result_df = pd.DataFrame(results)
    result_df.to_csv(results_path, index=False)
    return result_df


def is_clean_success(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def select_clean_evaluation_dataset(result_df, target_per_category, seed):
    if "clean_success" not in result_df.columns:
        raise ValueError("Simulation results do not contain a clean_success column.")

    clean_df = result_df[result_df["clean_success"].map(is_clean_success)].copy()
    selected = []
    for category in CATEGORIES:
        category_df = clean_df[clean_df["category"] == category]
        n_take = min(target_per_category, len(category_df))
        if n_take == 0:
            continue
        selected.append(category_df.sample(n=n_take, random_state=seed))

    if not selected:
        return pd.DataFrame(columns=result_df.columns)

    final_df = pd.concat(selected, ignore_index=True)
    final_df = final_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    final_df.insert(0, "evaluation_id", range(len(final_df)))
    return final_df


def print_counts(label, df):
    print("\n" + "=" * 70)
    print(label)
    print("=" * 70)
    if df.empty or "category" not in df.columns:
        print("No rows available.")
        return
    print(df["category"].value_counts().sort_index().to_string())


def print_existing_stats(paths):
    for key in ["all_combinations", "candidate_pool", "simulation_results", "evaluation_dataset"]:
        path = paths[key]
        if not path.exists():
            print(f"{path.name}: not found")
            continue
        df = pd.read_csv(path)
        print_counts(path.name, df)
        if "clean_success" in df.columns:
            clean_count = df["clean_success"].map(is_clean_success).sum()
            print(f"clean_success: {clean_count}/{len(df)}")


def confirm_or_exit(args, candidate_count):
    if args.yes or args.combinations_only or args.stats:
        return

    print(
        f"This will run up to {candidate_count} PyBullet simulations "
        "for the additional evaluation candidate pool."
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

    candidate_df = sample_by_category(
        all_df,
        per_category=args.pool_per_category,
        seed=args.seed,
    )
    candidate_df.to_csv(paths["candidate_pool"], index=False)
    print_counts("Sampled candidate pool", candidate_df)

    if args.combinations_only:
        print(f"\nWrote combinations to: {paths['root']}")
        return

    confirm_or_exit(args, len(candidate_df))
    result_df = run_candidate_simulations(candidate_df, paths, args)
    print_counts("Candidate simulation results", result_df)

    evaluation_df = select_clean_evaluation_dataset(
        result_df,
        target_per_category=args.target_per_category,
        seed=args.seed,
    )
    evaluation_df.to_csv(paths["evaluation_dataset"], index=False)
    print_counts("Selected additional evaluation dataset", evaluation_df)

    print(f"\nOutput directory: {paths['root']}")


if __name__ == "__main__":
    main()
