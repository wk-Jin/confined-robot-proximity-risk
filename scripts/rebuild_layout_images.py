"""Write any missing layout images of the generated dataset from the trial parameters.

Layout images are a deterministic function of the layout and trial parameters, so a
missing image can be regenerated with the LayoutImageGenerator used during data
generation. Existing images are left unchanged; a sample of them is regenerated and
compared byte for byte before anything is written.

Example:
    python scripts/rebuild_layout_images.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import franka_simulation as fs  # noqa: E402

LAYOUT = ["fence_x", "pillar_x", "pillar_y"]
TRIAL = ["pick_x_offset", "pick_y_offset", "place_x_offset", "place_y_offset"]


def render(row):
    generator = fs.LayoutImageGenerator()
    generator.generate({k: float(row[k]) for k in LAYOUT}, {k: float(row[k]) for k in TRIAL})
    return generator.image.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data/generated_dataset"))
    parser.add_argument("--all-trials", action="store_true",
                        help="Also write images for trials that are not clean successes.")
    parser.add_argument("--check", type=int, default=200, help="Existing images to verify first.")
    args = parser.parse_args()

    df = pd.read_csv(args.data_dir / "trial_summary_all.csv")
    if not args.all_trials:
        df = df[df["clean_success"].astype(str).str.strip().str.lower().isin(["1", "1.0", "true"])]
    image_dir = args.data_dir / "layout_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    stems = df["layout_image_path"].astype(str).str.replace("\\", "/").str.split("/").str[-1].str.rsplit(".", n=1).str[0]
    paths = [image_dir / f"{s}.npy" for s in stems]
    exists = np.array([p.exists() for p in paths])

    rows = df.reset_index(drop=True)
    for i in np.flatnonzero(exists)[: args.check]:
        if not np.array_equal(np.load(paths[i]), render(rows.loc[i])):
            raise SystemExit(f"Regenerated image differs from the existing file {paths[i]}; nothing written.")
    print(f"Verified {min(args.check, int(exists.sum()))} existing images.")

    missing = np.flatnonzero(~exists)
    for i in missing:
        np.save(paths[i], render(rows.loc[i]))
    print(f"Rows: {len(rows)}, existing: {int(exists.sum())}, written: {len(missing)}")


if __name__ == "__main__":
    main()
