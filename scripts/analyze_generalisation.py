"""
Analyses of the C0-C4 additional evaluation that are reported in the paper.

No PyBullet simulation and no model training is run. Layout images for the standard
test set are generated from the trial parameters with the LayoutImageGenerator used
for the datasets.

Outputs (CSV, written to --output-dir):
    c0_grid_interpolation.csv     C0 versus the grid-interpolation reference built from
                                  clean-success trials at the surrounding grid corners (3.9, 4.3.1)
    standard_test_by_layout.csv   Standard-test MAE by held-out layout                    (4.3.1)
    category_composition.csv      Share of samples with each extrapolated group, per category (4.4.2)
    fence_clean_rate.csv          Clean-success rate below / within / above the fence range (3.8)

Example:
    python scripts/analyze_generalisation.py
"""

import argparse
import itertools
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

CATEGORIES = ["C0", "C1", "C2", "C3", "C4"]
MODELS = ["resnet18", "vit_tiny"]
PARAMS = [
    "fence_x",
    "pillar_x",
    "pillar_y",
    "pick_x_offset",
    "pick_y_offset",
    "place_x_offset",
    "place_y_offset",
]
TRIAL_PARAMS = PARAMS[3:]
GROUP_FLAGS = ["fence_extrap", "pillar_extrap", "pick_extrap", "place_extrap"]

# Model-development ranges (Table 2) and grid steps.
DEV_RANGE = {
    "fence_x": (0.350, 0.500),
    "pillar_x": (-0.45, -0.25),
    "pillar_y": (-0.55, -0.35),
    "pick_x_offset": (-0.10, 0.10),
    "pick_y_offset": (-0.10, 0.10),
    "place_x_offset": (-0.10, 0.10),
    "place_y_offset": (-0.10, 0.10),
}
TOL = 1e-6


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/additional_evaluation_dataset"))
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/additional_evaluation"))
    parser.add_argument("--training-csv", type=Path, default=Path("data/generated_dataset/trial_summary_all.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/multi_seed_training"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/generalisation_analysis"))
    parser.add_argument("--train-seeds", default="42,43,44")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--skip-standard-test", action="store_true")
    return parser.parse_args()


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def as_bool(series):
    return series.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes"])


def load_inputs(args, seeds):
    data_csv = args.data_dir / "additional_evaluation_dataset.csv"
    eval_df = pd.read_csv(data_csv)
    eval_df["key"] = eval_df["layout_image_path"].astype(str).str.replace("\\", "/").str.split("/").str[-1]
    for flag in GROUP_FLAGS:
        eval_df[flag] = as_bool(eval_df[flag])
    if not as_bool(eval_df["clean_success"]).all():
        raise SystemExit("The evaluation dataset contains non-clean-success rows.")

    pred = {}
    for seed in seeds:
        path = args.evaluation_dir / f"seed_{seed}" / "all_predictions.csv"
        p = pd.read_csv(path)
        p["key"] = p["layout_image_path"].astype(str).str.replace("\\", "/").str.split("/").str[-1]
        merged = eval_df[["key"]].merge(p[["key", "resnet18_pred", "vit_tiny_pred", "true_value"]], on="key", how="left")
        if merged.isna().any().any() or len(merged) != len(eval_df):
            raise SystemExit(f"Predictions in {path} do not match the evaluation dataset.")
        if not np.allclose(merged["true_value"].to_numpy(), eval_df["invaded_grid_count"].to_numpy()):
            raise SystemExit(f"True values in {path} differ from the evaluation dataset.")
        pred[seed] = {m: merged[f"{m}_pred"].to_numpy(dtype=float) for m in MODELS}

    training = pd.read_csv(args.training_csv)
    sim_path = args.data_dir / "candidate_simulation_results.csv"
    sim = pd.read_csv(sim_path) if sim_path.exists() else None
    return eval_df, pred, training, sim


def grid_interpolation_baseline(eval_df, pred, seeds, training):
    # Use clean-success corner trials only, matching the population of the C0 targets.
    training = training[as_bool(training["clean_success"])]
    lookup = {}
    for row in training[PARAMS + ["invaded_grid_count"]].itertuples(index=False):
        key = tuple(round(float(v), 4) for v in row[:-1])
        lookup.setdefault(key, row[-1])
    # C0 values lie midway between grid levels: half of the Table 2 grid step.
    half = {"fence_x": 0.0125, "pillar_x": 0.05, "pillar_y": 0.05, **{p: 0.025 for p in TRIAL_PARAMS}}

    c0 = eval_df.index[eval_df["category"] == "C0"].to_numpy()
    interp, n_corner = [], []
    for i in c0:
        r = eval_df.loc[i]
        values = []
        for signs in itertools.product([-1, 1], repeat=len(PARAMS)):
            key = tuple(round(float(r[p]) + s * half[p], 4) for p, s in zip(PARAMS, signs))
            if key in lookup:
                values.append(lookup[key])
        interp.append(np.mean(values) if values else np.nan)
        n_corner.append(len(values))
    interp = np.asarray(interp)
    y = eval_df.loc[c0, "invaded_grid_count"].to_numpy(dtype=float)
    base_err = np.abs(interp - y)
    rows = [{"method": "grid_interpolation", "n": len(c0), "mae": float(np.nanmean(base_err)),
             "mae_seed_sd": 0.0, "share_better_than_baseline": np.nan,
             "mean_corners_used": float(np.mean(n_corner)), "min_corners_used": int(np.min(n_corner))}]
    for model in MODELS:
        maes, better = [], []
        for s in seeds:
            err = np.abs(pred[s][model][c0] - y)
            maes.append(err.mean())
            better.append(100 * np.mean(err < base_err))
        rows.append({"method": model, "n": len(c0), "mae": float(np.mean(maes)),
                     "mae_seed_sd": float(np.std(maes, ddof=1)),
                     "share_better_than_baseline": float(np.mean(better)),
                     "mean_corners_used": np.nan, "min_corners_used": np.nan})
    return pd.DataFrame(rows)


def setup_models(args):
    import run_additional_evaluation as rae
    rae.load_libraries(require_timm=True)
    import franka_simulation as fs
    return rae, fs


def generate_image(fs, layout, trial):
    generator = fs.LayoutImageGenerator()
    return generator.generate(layout, trial).copy()


def predict_images(rae, args, images, seeds, models):
    out = {}
    transforms_pair = rae.make_preprocess_transform(224)
    model_args = types.SimpleNamespace(hidden_dim=args.hidden_dim, dropout=args.dropout)
    for model_name in models:
        for seed in seeds:
            path = rae.find_model_path(rae.find_seed_model_dir(args.model_dir, seed), model_name)
            model, mean, std = rae.load_trained_model(model_name, path, model_args, args.device)
            out[(model_name, seed)] = rae.predict_model(
                model, images, args.batch_size, args.device, mean, std, transforms_pair
            )
            log(f"  predicted {len(images)} images with {model_name} seed {seed}")
    return out


def standard_test_by_layout(args, rae, fs, seeds, training):
    import run_multi_seed_training as rmt
    rmt.load_libraries(require_timm=False)
    clean = training[rmt.clean_success_mask(training["clean_success"])].copy()
    _, _, test_df, _, _, test_layouts = rmt.split_by_layout(clean, "layout_id", args.split_seed)
    test_df = test_df.reset_index(drop=True)
    log(f"Standard test: {len(test_df)} samples from {len(test_layouts)} layouts")
    images = [
        generate_image(fs,
                       {"fence_x": r.fence_x, "pillar_x": r.pillar_x, "pillar_y": r.pillar_y},
                       {p: getattr(r, p) for p in TRIAL_PARAMS})
        for r in test_df.itertuples()
    ]
    preds = predict_images(rae, args, images, seeds, MODELS)
    y = test_df["invaded_grid_count"].to_numpy(dtype=float)

    rows = []
    groups = [("all", np.arange(len(test_df)))] + [
        (int(l), np.where(test_df["layout_id"].to_numpy() == l)[0]) for l in sorted(test_layouts)
    ]
    for layout, idx in groups:
        first = test_df.iloc[idx[0]]
        row = {"layout_id": layout, "n": len(idx)}
        if layout != "all":
            row.update({"fence_x": first.fence_x, "pillar_x": first.pillar_x, "pillar_y": first.pillar_y})
        for model in MODELS:
            maes = [np.mean(np.abs(preds[(model, s)][idx] - y[idx])) for s in seeds]
            row[f"{model}_mae_mean"] = float(np.mean(maes))
            row[f"{model}_mae_seed_sd"] = float(np.std(maes, ddof=1))
        rows.append(row)

    check = []
    for s in seeds:
        result_path = rae.find_seed_model_dir(args.model_dir, s) / "test_results.json"
        if result_path.exists():
            reported = json.loads(result_path.read_text())["cnn"]["mae"]
            reproduced = float(np.mean(np.abs(preds[("resnet18", s)] - y)))
            check.append(f"seed {s}: reported {reported:.4f}, reproduced {reproduced:.4f}")
    for line in check:
        log(f"  standard-test check {line}")
    return pd.DataFrame(rows)


def composition(eval_df, sim):
    rows = []
    summary_rows = None
    for category in CATEGORIES:
        d = eval_df[eval_df["category"] == category]
        row = {"category": category, "n": len(d)}
        for flag in GROUP_FLAGS:
            row[f"share_{flag}"] = float(d[flag].mean())
        row["mean_actual"] = float(d["invaded_grid_count"].mean())
        row["sd_actual"] = float(d["invaded_grid_count"].std())
        if sim is not None:
            s = sim[sim["category"] == category]
            row["n_simulated"] = len(s)
            row["clean_rate"] = float(as_bool(s["clean_success"]).mean()) if len(s) else np.nan
        rows.append(row)
    comp = pd.DataFrame(rows)

    if sim is not None:
        lo, hi = DEV_RANGE["fence_x"]
        cls = np.where(sim["fence_x"] < lo - TOL, "below_range",
                       np.where(sim["fence_x"] > hi + TOL, "above_range", "within_range"))
        clean = as_bool(sim["clean_success"])
        summary_rows = (
            pd.DataFrame({"fence_position": cls, "clean": clean})
            .groupby("fence_position")["clean"].agg(n_simulated="size", clean_rate="mean").reset_index()
        )
    return comp, summary_rows


def main():
    args = parse_args()
    seeds = [int(s) for s in args.train_seeds.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir

    eval_df, pred, training, sim = load_inputs(args, seeds)
    log(f"Loaded {len(eval_df)} evaluation samples and predictions for seeds {seeds}")

    grid_interpolation_baseline(eval_df, pred, seeds, training).to_csv(out / "c0_grid_interpolation.csv", index=False)
    log("C0 grid-interpolation reference")

    comp, fence_clean = composition(eval_df, sim)
    comp.to_csv(out / "category_composition.csv", index=False)
    if fence_clean is not None:
        fence_clean.to_csv(out / "fence_clean_rate.csv", index=False)
    log("Category composition and fence clean-success rates")

    if not args.skip_standard_test:
        rae, fs = setup_models(args)
        standard_test_by_layout(args, rae, fs, seeds, training).to_csv(out / "standard_test_by_layout.csv", index=False)
        log("Standard test by layout")

    log(f"Done. Outputs in {out}")


if __name__ == "__main__":
    main()
