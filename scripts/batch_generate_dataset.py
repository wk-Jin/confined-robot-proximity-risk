"""
Batch data generation for the confined-workspace Franka Panda dataset.

This script enumerates the layout and trial parameter combinations reported in
Table 2 of the paper, repeatedly calls `run_simulation()`, and appends the
resulting trial-level summaries to a dataset CSV. It can also save per-step
records and 128x128 RGB layout images for model training.

Outputs:
    - trial_summary_all.csv: one row per simulated trial.
    - step_data_all.csv: optional per-step records for all trials.
    - layout_images/*.npy: one layout image per trial.

Example:
    python batch_generate_dataset.py --yes
    python batch_generate_dataset.py --stats
"""

import argparse
import importlib
import os
import sys
import csv
import time
import itertools
from pathlib import Path
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent if SCRIPT_DIR.name == "scripts" else SCRIPT_DIR

for import_path in (SCRIPT_DIR, PROJECT_ROOT, PROJECT_ROOT / "src"):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)


def load_simulation_module():
    """Load the simulation module used by the batch generator."""
    explicit_name = os.environ.get("FRANKA_SIMULATION_MODULE")
    candidates = (
        [explicit_name]
        if explicit_name
        else [
            "franka_simulation",
            "src.franka_simulation",
        ]
    )

    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            if exc.name == module_name:
                continue
            raise

        return module

    searched = ", ".join(candidates)
    raise ModuleNotFoundError(
        "Could not import a simulation module. Expected one of: "
        f"{searched}. Set FRANKA_SIMULATION_MODULE to override."
    )


SIMULATION_MODULE = None
RUN_SIMULATION = None
LAYOUT_IMAGE_GENERATOR = None


def get_simulation_api():
    """Return the simulation entry point and layout-image generator class."""
    global SIMULATION_MODULE
    global RUN_SIMULATION
    global LAYOUT_IMAGE_GENERATOR

    if SIMULATION_MODULE is None:
        SIMULATION_MODULE = load_simulation_module()
        RUN_SIMULATION = SIMULATION_MODULE.run_simulation
        LAYOUT_IMAGE_GENERATOR = SIMULATION_MODULE.LayoutImageGenerator

    return RUN_SIMULATION, LAYOUT_IMAGE_GENERATOR


# ============================================================
# Dataset generation settings
# ============================================================

# Use DIRECT mode by default for faster unattended generation.
HEADLESS = True

# Layout parameters
# Values follow the model-development parameter grid reported in Table 2.
FENCE_X_VALUES   = [0.350, 0.375, 0.400, 0.425, 0.450, 0.475, 0.500]

PILLAR_X_VALUES  = [-0.45, -0.35, -0.25]
PILLAR_Y_VALUES  = [-0.55, -0.45, -0.35]

# Trial parameters
PICK_X_OFFSETS   = [-0.10, -0.05, 0.00, 0.05, 0.10]
PICK_Y_OFFSETS   = [-0.10, -0.05, 0.00, 0.05, 0.10]

PLACE_X_OFFSETS  = [-0.10, -0.05, 0.00, 0.05, 0.10]
PLACE_Y_OFFSETS  = [-0.10, -0.05, 0.00, 0.05, 0.10]

# Number of trials: 5 x 5 x 5 x 5 = 625 per layout.
# Total simulations: 63 x 625 = 39,375.

# Output files
OUTPUT_DIR              = Path(
    os.environ.get(
        "FRANKA_DATASET_OUTPUT_DIR",
        str(PROJECT_ROOT / "outputs" / "generated_dataset"),
    )
).expanduser().resolve()
ALL_STEP_DATA_PATH      = OUTPUT_DIR / "step_data_all.csv"
ALL_TRIAL_SUMMARY_PATH  = OUTPUT_DIR / "trial_summary_all.csv"
LAYOUT_IMAGES_DIR       = OUTPUT_DIR / "layout_images"
PROGRESS_LOG_PATH       = OUTPUT_DIR / "progress.log"

# Runtime options
SAVE_STEP_DATA   = False
RESUME           = True
PRINT_EVERY      = 5


def ensure_parent_dir(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def configure_output_dir(output_dir):
    """Update all output paths from a user-provided root directory."""
    global OUTPUT_DIR
    global ALL_STEP_DATA_PATH
    global ALL_TRIAL_SUMMARY_PATH
    global LAYOUT_IMAGES_DIR
    global PROGRESS_LOG_PATH

    OUTPUT_DIR = Path(output_dir).expanduser().resolve()
    ALL_STEP_DATA_PATH = OUTPUT_DIR / "step_data_all.csv"
    ALL_TRIAL_SUMMARY_PATH = OUTPUT_DIR / "trial_summary_all.csv"
    LAYOUT_IMAGES_DIR = OUTPUT_DIR / "layout_images"
    PROGRESS_LOG_PATH = OUTPUT_DIR / "progress.log"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the PyBullet simulation dataset used by the paper."
    )
    parser.add_argument(
        "--output-dir",
        default=str(OUTPUT_DIR),
        help="Directory for generated CSV files, logs, and layout images.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Start generation without an interactive confirmation prompt.",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print summary statistics for an existing generated dataset.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Run simulations with the PyBullet GUI instead of DIRECT mode.",
    )
    parser.add_argument(
        "--save-step-data",
        action="store_true",
        help="Also save per-step records. This can produce a very large CSV.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip completed (layout_id, trial_id) pairs.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=PRINT_EVERY,
        help="Print progress every N newly completed simulations.",
    )
    return parser.parse_args()


# ============================================================
# Helper functions
# ============================================================

def generate_layout_list():
    """Generate all layout parameter combinations."""
    layouts = []
    layout_id = 0
    for fence_x, pillar_x, pillar_y in itertools.product(
            FENCE_X_VALUES, PILLAR_X_VALUES, PILLAR_Y_VALUES):
        layouts.append({
            'layout_id': layout_id,
            'fence_x':   fence_x,
            'pillar_x':  pillar_x,
            'pillar_y':  pillar_y,
        })
        layout_id += 1
    return layouts


def generate_trial_list():
    """Generate all trial parameter combinations."""
    trials = []
    trial_id = 0
    for pxo, pyo, plxo, plyo in itertools.product(
            PICK_X_OFFSETS, PICK_Y_OFFSETS,
            PLACE_X_OFFSETS, PLACE_Y_OFFSETS):
        trials.append({
            'trial_id':       trial_id,
            'pick_x_offset':  pxo,
            'pick_y_offset':  pyo,
            'place_x_offset': plxo,
            'place_y_offset': plyo,
        })
        trial_id += 1
    return trials


def get_completed_set(trial_summary_path):
    """Return completed (layout_id, trial_id) pairs for resume mode."""
    completed = set()
    if not os.path.exists(trial_summary_path):
        return completed
    try:
        with open(trial_summary_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                lid = int(row['layout_id'])
                tid = int(row['trial_id'])
                completed.add((lid, tid))
    except Exception as e:
        print(f"  Failed to read resume state: {e}")
    return completed


def append_step_data(step_logger, layout_id, trial_id, output_path):
    """Append per-step records from one trial to the combined CSV."""
    if not step_logger.records:
        return

    for r in step_logger.records:
        r['layout_id'] = layout_id
        r['trial_id'] = trial_id

    keys = list(step_logger.records[0].keys())
    for k in ['trial_id', 'layout_id']:
        if k in keys:
            keys.remove(k)
            keys.insert(0, k)

    file_exists = os.path.exists(output_path)
    mode = 'a' if file_exists else 'w'
    ensure_parent_dir(output_path)
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerows(step_logger.records)


def append_trial_summary(result, output_path):
    """Append one trial-level summary row to the combined CSV."""
    row = {k: v for k, v in result.items() if not k.startswith('_')}
    keys = list(row.keys())

    file_exists = os.path.exists(output_path)
    mode = 'a' if file_exists else 'w'
    ensure_parent_dir(output_path)
    with open(output_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def format_time(seconds):
    """Format seconds as a compact human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def log_progress(message):
    """Log progress to both the console and progress.log."""
    print(message)
    try:
        ensure_parent_dir(PROGRESS_LOG_PATH)
        with open(PROGRESS_LOG_PATH, "a", encoding="utf-8") as f:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    except Exception:
        pass


# ============================================================
# Main batch-generation routine
# ============================================================

def main(require_confirmation=True, headless=HEADLESS, save_step_data=SAVE_STEP_DATA,
         resume=RESUME, print_every=PRINT_EVERY):
    run_simulation, layout_image_generator = get_simulation_api()

    # Create output directories.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LAYOUT_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # Generate the full layout x trial parameter grid.
    layouts = generate_layout_list()
    trials  = generate_trial_list()
    n_layouts = len(layouts)
    n_trials = len(trials)
    n_total = n_layouts * n_trials

    # Resume mode skips trials already present in trial_summary_all.csv.
    completed = get_completed_set(ALL_TRIAL_SUMMARY_PATH) if resume else set()
    n_remaining = n_total - len(completed)

    print("\n" + "="*70)
    print("Batch dataset generation")
    print("="*70)
    print("  Layout parameters:")
    print(f"    fence_x  : {FENCE_X_VALUES} ({len(FENCE_X_VALUES)} values)")
    print(f"    pillar_x : {PILLAR_X_VALUES} ({len(PILLAR_X_VALUES)} values)")
    print(f"    pillar_y : {PILLAR_Y_VALUES} ({len(PILLAR_Y_VALUES)} values)")
    print(f"    layouts  : {n_layouts}")
    print("  Trial parameters:")
    print(f"    pick_offset  : {len(PICK_X_OFFSETS)} x {len(PICK_Y_OFFSETS)}")
    print(f"    place_offset : {len(PLACE_X_OFFSETS)} x {len(PLACE_Y_OFFSETS)}")
    print(f"    trials per layout: {n_trials}")
    print("  " + "-"*28)
    print(f"  total simulations: {n_total}")
    if completed:
        print(f"  completed: {len(completed)}")
        print(f"  remaining: {n_remaining}")
    print(f"  HEADLESS = {headless}")
    print("="*70)

    avg_sim_time = 3.0 if headless else 12.0
    estimated = n_remaining * avg_sim_time
    print(f"  estimated time: {format_time(estimated)} "
          f"(assuming {avg_sim_time:.0f}s per simulation)")
    print()

    if n_remaining == 0:
        print("  All simulations are already complete.")
        return

    if require_confirmation:
        try:
            ans = input("Continue? (y/n): ").strip().lower()
            if ans != 'y':
                print("Cancelled.")
                return
        except KeyboardInterrupt:
            print("\nCancelled.")
            return

    log_progress(f"Started: {n_remaining} simulations remaining")
    start_time = time.time()
    completed_count = 0
    failed_count = 0

    for layout in layouts:
        for trial in trials:
            sim_id = (layout['layout_id'], trial['trial_id'])

            if sim_id in completed:
                continue

            try:
                t_sim_start = time.time()

                result = run_simulation(
                    layout_params=layout,
                    trial_params=trial,
                    headless=headless,
                    verbose=False,
                    save_step_data=False,
                    save_layout_image=False,
                )

                t_sim_elapsed = time.time() - t_sim_start

                image_name = (
                    f"layout_{layout['layout_id']:03d}_"
                    f"trial_{trial['trial_id']:03d}.npy"
                )
                img_path = LAYOUT_IMAGES_DIR / image_name
                img_gen = layout_image_generator()
                img_gen.generate(layout, trial)
                img_gen.save(str(img_path))

                result['layout_image_path'] = (Path("layout_images") / image_name).as_posix()

                if save_step_data and '_step_logger' in result:
                    append_step_data(
                        result['_step_logger'],
                        layout['layout_id'], trial['trial_id'],
                        ALL_STEP_DATA_PATH)

                append_trial_summary(result, ALL_TRIAL_SUMMARY_PATH)

                completed_count += 1

                if print_every > 0 and completed_count % print_every == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed_count
                    remaining = (n_remaining - completed_count) * avg_time

                    msg = (f"  [{completed_count:>4}/{n_remaining}] "
                           f"layout={layout['layout_id']:>3} "
                           f"trial={trial['trial_id']:>3} | "
                           f"sim={t_sim_elapsed:.1f}s | "
                           f"avg={avg_time:.1f}s | "
                           f"remaining={format_time(remaining)} | "
                           f"invaded={result['invaded_grid_count']:>4} "
                           f"success={result['task_success']}")
                    log_progress(msg)

            except Exception as e:
                failed_count += 1
                msg = (f"  Failed layout={layout['layout_id']} "
                       f"trial={trial['trial_id']}: {e}")
                log_progress(msg)
                continue

    total_elapsed = time.time() - start_time
    print("\n" + "="*70)
    print("Dataset generation complete")
    print("="*70)
    print(f"  completed: {completed_count}")
    print(f"  failed: {failed_count}")
    print(f"  elapsed: {format_time(total_elapsed)}")
    print(f"  average simulation time: {total_elapsed/max(completed_count,1):.2f}s")
    print()
    print(f"  output directory: {OUTPUT_DIR}/")
    print(f"     trial_summary_all.csv ({n_total} expected rows)")
    if save_step_data:
        print(f"     step_data_all.csv     (~{n_total*2500:,} expected rows)")
    print(f"     layout_images/        ({n_total} expected NPY files)")
    print("="*70)
    log_progress(f"Completed: {completed_count}/{n_remaining}, "
                 f"failed {failed_count}, elapsed {format_time(total_elapsed)}")


# ============================================================
# Dataset statistics
# ============================================================

def print_dataset_stats():
    """Print compact statistics for a generated dataset."""
    if not os.path.exists(ALL_TRIAL_SUMMARY_PATH):
        print("No generated dataset was found.")
        return

    try:
        rows = []
        with open(ALL_TRIAL_SUMMARY_PATH, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        print(f"Failed to read dataset: {e}")
        return

    if not rows:
        print("The dataset file is empty.")
        return

    print("\n" + "="*70)
    print("Dataset statistics")
    print("="*70)
    print(f"  total trials: {len(rows)}")

    success = sum(1 for r in rows if int(r.get('task_success', 0)) == 1)
    print(f"  task success: {success}/{len(rows)} ({success/len(rows)*100:.1f}%)")

    invaded = [int(r['invaded_grid_count']) for r in rows]
    hazard_ratios = [float(r.get('hazard_time_ratio', 0)) for r in rows]
    intrusions = [float(r.get('max_intrusion_depth_mm', 0)) for r in rows]

    print("  invaded_grid_count:")
    print(f"    mean {np.mean(invaded):.1f}, min {min(invaded)}, "
          f"max {max(invaded)}")
    print(f"  hazard_time_ratio:")
    print(f"    mean {np.mean(hazard_ratios):.3f}, "
          f"min {min(hazard_ratios):.3f}, max {max(hazard_ratios):.3f}")
    print(f"  max_intrusion_depth_mm:")
    print(f"    mean {np.mean(intrusions):.1f}, "
          f"min {min(intrusions):.1f}, max {max(intrusions):.1f}")

    from collections import defaultdict
    by_layout = defaultdict(list)
    for r in rows:
        by_layout[int(r['layout_id'])].append(float(r.get('hazard_time_ratio', 0)))

    print(f"\n  top 5 layouts by mean hazard_time_ratio:")
    layout_means = [(lid, np.mean(vals)) for lid, vals in by_layout.items()]
    layout_means.sort(key=lambda x: x[1], reverse=True)
    for lid, mean_val in layout_means[:5]:
        print(f"    layout {lid:>3}: {mean_val:.3f}")

    print("="*70)


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    configure_output_dir(args.output_dir)

    if args.stats:
        print_dataset_stats()
    else:
        main(
            require_confirmation=not args.yes,
            headless=not args.gui,
            save_step_data=args.save_step_data,
            resume=not args.no_resume,
            print_every=args.print_every,
        )
        print()
        print_dataset_stats()
