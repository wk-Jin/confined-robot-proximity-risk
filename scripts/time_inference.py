"""Time layout-image generation and model inference per candidate (Section 4.6).

Each candidate is processed one at a time: layout image generation, preprocessing and
prediction with the seed-42 checkpoints. The simulation time is the mean of the
sim_time_s column recorded while generating the additional evaluation dataset.

Example:
    python scripts/time_inference.py
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_multi_seed_training as rmt, franka_simulation as fs

ap = argparse.ArgumentParser()
ap.add_argument("--data-csv", default="data/additional_evaluation_dataset/additional_evaluation_dataset.csv")
ap.add_argument("--model-dir", default="outputs/multi_seed_training/seed_42")
ap.add_argument("--n", type=int, default=200)
a = ap.parse_args()

rmt.load_libraries(require_timm=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ev = pd.read_csv(a.data_csv).head(a.n)
tf = rmt.make_resize_transform(224)

def make_image(r):
    g = fs.LayoutImageGenerator()
    g.generate({"fence_x": r.fence_x, "pillar_x": r.pillar_x, "pillar_y": r.pillar_y},
               {k: getattr(r, k) for k in ["pick_x_offset", "pick_y_offset", "place_x_offset", "place_y_offset"]})
    return g.image.copy()

def prep(img):
    return tf(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0).unsqueeze(0).to(dev)

rows = list(ev.itertuples(index=False))
for name in ["resnet18", "vit_tiny"]:
    ck = torch.load(Path(a.model_dir) / f"{name}_layout_risk.pt", map_location="cpu", weights_only=False)
    m = rmt.build_model(name, 128, 0.3, False)
    m.load_state_dict({k.replace("backbone.", "", 1): v for k, v in ck["model_state_dict"].items()})
    m.to(dev).eval()
    with torch.no_grad():
        for r in rows[:5]:
            m(prep(make_image(r)))                      # warm-up
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for r in rows:                                  # image generation + preprocessing + prediction, one candidate at a time
            m(prep(make_image(r)))
        if dev.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / len(rows)
    print(f"{name}: {dt * 1000:.1f} ms per candidate on {dev}")

sim = pd.read_csv(Path(a.data_csv).parent / "candidate_simulation_results.csv")
print(f"simulation: {sim.sim_time_s.mean():.1f} s per trial (mean of {len(sim)} trials)")
