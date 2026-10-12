"""Regenerate Fig. 11 (Score-CAM maps for the three representative cases).

Procedure (Section 3.10 of the paper):
  * Representative seed: the training seed whose overall additional-evaluation metrics
    (MAE, RMSE, R2, ME and UPR for both models) have the smallest summed standardised
    deviation from the three-seed means.
  * Cases, selected from errors averaged over the three seeds:
      Case 1  smallest combined absolute error of the two models
      Case 2  largest ViT-Tiny minus ResNet-18 absolute error
      Case 3  most negative combined signed error among samples in the top 25% of actual values
  * Score-CAM maps are computed with the representative-seed models only.

Example:
    python scripts/make_scorecam_figure.py
"""

import argparse
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models

TRAIN_SEEDS = [42, 43, 44]
METRIC_COLS = ["mae", "rmse", "r2", "me", "upr"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/additional_evaluation"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/additional_evaluation_dataset"))
    parser.add_argument("--model-dir", type=Path, default=Path("outputs/multi_seed_training"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scorecam"))
    return parser.parse_args()


# ---------------------------------------------------------------- selection

def select_representative_seed(evaluation_dir, output_dir):
    overall = pd.read_csv(evaluation_dir / "summary_overall_by_seed.csv")
    overall["model"] = overall["model"].map({"resnet18": "cnn", "vit_tiny": "vit"})
    pivot = overall.pivot(index="train_seed", columns="model", values=METRIC_COLS)
    pivot.columns = [f"{m}_{metric}" for metric, m in pivot.columns]
    pivot = pivot.reindex(TRAIN_SEEDS)
    z = ((pivot - pivot.mean()) / pivot.std(ddof=0)).fillna(0.0)
    deviation = z.abs().sum(axis=1)
    table = pivot.copy()
    table["deviation_score"] = deviation
    table.to_csv(output_dir / "representative_seed_selection.csv")
    print(table.round(4))
    return int(deviation.idxmin())


def load_predictions(evaluation_dir):
    dfs = {}
    for s in TRAIN_SEEDS:
        df = pd.read_csv(evaluation_dir / f"seed_{s}" / "all_predictions.csv")
        dfs[s] = df.rename(columns={"resnet18_pred": "cnn_pred", "resnet18_error": "cnn_error",
                                    "vit_tiny_pred": "vit_pred", "vit_tiny_error": "vit_error"})
    keys = dfs[TRAIN_SEEDS[0]]["layout_image_path"].values
    true = dfs[TRAIN_SEEDS[0]]["true_value"].values
    for s in TRAIN_SEEDS[1:]:
        assert (dfs[s]["layout_image_path"].values == keys).all(), f"seed {s}: row order differs"
        assert np.allclose(dfs[s]["true_value"].values, true), f"seed {s}: true values differ"
    return dfs


def select_cases(dfs):
    avg = dfs[TRAIN_SEEDS[0]][["category", "true_value"]].copy()
    for key in ("cnn", "vit"):
        avg[f"{key}_error"] = np.mean([dfs[s][f"{key}_error"].values for s in TRAIN_SEEDS], axis=0)
        avg[f"{key}_abs_error"] = np.mean([np.abs(dfs[s][f"{key}_error"].values) for s in TRAIN_SEEDS], axis=0)
    avg["combined_abs_error"] = avg["cnn_abs_error"] + avg["vit_abs_error"]
    avg["divergence"] = avg["vit_abs_error"] - avg["cnn_abs_error"]
    avg["combined_error"] = avg["cnn_error"] + avg["vit_error"]

    case1 = avg.sort_values("combined_abs_error").iloc[0]
    case2 = avg.sort_values("divergence", ascending=False).iloc[0]
    high = avg[avg["true_value"] >= avg["true_value"].quantile(0.75)]
    case3 = high.sort_values("combined_error").iloc[0]

    src = dfs[TRAIN_SEEDS[0]]
    flags = [("fence_extrap", "Fence"), ("pillar_extrap", "Pillar"),
             ("pick_extrap", "Pick"), ("place_extrap", "Place")]
    cases = []
    for label, row in zip(["Case 1", "Case 2", "Case 3"], [case1, case2, case3]):
        s = src.loc[row.name]
        names = [n for col, n in flags if str(s[col]).strip().lower() in ("true", "1", "1.0")]
        cases.append({"label": label, "sample_index": row.name, "category": row["category"],
                      "true_value": float(row["true_value"]), "extrap_vars": ", ".join(names) or "None",
                      "layout_image_path": s["layout_image_path"]})
    return cases


# ---------------------------------------------------------------- models

class ResNet18RiskRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.backbone(x).squeeze(-1)


class ViTTinyRiskRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.backbone = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=0)
        self.head = nn.Sequential(nn.Linear(self.backbone.num_features, 128), nn.ReLU(), nn.Dropout(0.3),
                                  nn.Linear(128, 1))

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


def load_model(cls, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = cls().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt["target_mean"], ckpt["target_std"]


def seed_model_dir(model_dir, seed):
    for name in (f"seed_{seed}", f"seed{seed}"):
        if (model_dir / name).is_dir():
            return model_dir / name
    raise FileNotFoundError(f"No checkpoint folder for seed {seed} under {model_dir}")


def load_and_preprocess(image_dir, image_path, device):
    img = np.load(image_dir / os.path.basename(str(image_path).replace("\\", "/")))
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    raw = img.copy()
    x = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1).unsqueeze(0)
    x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    x = (x - torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return x.to(device), raw


# ---------------------------------------------------------------- Score-CAM

def _score_cam(model, input_tensor, feature_map, device):
    """feature_map: (1, C, h, w). Each channel, upsampled and min-max normalised, masks the input."""
    up = torch.nn.functional.interpolate(feature_map, size=(224, 224), mode="bilinear", align_corners=False)
    cam = torch.zeros(224, 224, device=device)
    with torch.no_grad():
        for i in range(up.shape[1]):
            fmap = up[0, i]
            lo, hi = fmap.min(), fmap.max()
            if hi - lo < 1e-8:
                continue
            score = model(input_tensor * ((fmap - lo) / (hi - lo)).unsqueeze(0).unsqueeze(0)).item()
            cam += score * fmap
    cam = torch.relu(cam)
    lo, hi = cam.min(), cam.max()
    if hi - lo > 1e-8:
        cam = (cam - lo) / (hi - lo)
    return cam.cpu().numpy()


def _capture(model, layer, input_tensor):
    acts = []
    handle = layer.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    with torch.no_grad():
        model(input_tensor)
    handle.remove()
    return acts[0]


def score_cam_resnet(model, input_tensor, device):
    return _score_cam(model, input_tensor, _capture(model, model.backbone.layer4[-1], input_tensor), device)


def score_cam_vit(model, input_tensor, device):
    acts = _capture(model, model.backbone.blocks[-1], input_tensor)      # (1, 197, 192)
    patches = acts[:, 1:, :]                                             # drop the CLS token
    n = int(np.sqrt(patches.shape[1]))                                   # 14
    fmap = patches.reshape(1, n, n, patches.shape[-1]).permute(0, 3, 1, 2)
    return _score_cam(model, input_tensor, fmap, device)


# ---------------------------------------------------------------- figure

def _rgb(raw):
    img = cv2.resize(raw.copy(), (224, 224), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    return np.clip(img / 255.0 if img.max() > 1.5 else img, 0, 1)


def _norm(cam):
    cam = cv2.resize(np.asarray(cam, dtype=np.float32), (224, 224), interpolation=cv2.INTER_LINEAR)
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


def plot_scorecam_3x3(results, save_path, cmap="jet", overlay_alpha=0.48):
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 9.6), facecolor="white")
    for j, title in enumerate(["Input", "ResNet-18 Score-CAM", "ViT-Tiny Score-CAM"]):
        axes[0, j].set_title(title, fontsize=12, fontweight="bold", pad=8)
    im = None
    for i, r in enumerate(results):
        c = r["case"]
        raw = _rgb(r["raw"])
        axes[i, 0].imshow(raw)
        axes[i, 0].set_xlabel(f"Case {c['label'].split()[-1]} | {str(c['category']).split('_')[0]} | "
                              f"Actual = {c['true_value']:.0f}\nExtrapolated: {c['extrap_vars']}",
                              fontsize=11, labelpad=6)
        for j, key in ((1, "cnn"), (2, "vit")):
            axes[i, j].imshow(raw)
            im = axes[i, j].imshow(_norm(r[f"{key}_cam"]), cmap=cmap, alpha=overlay_alpha, vmin=0, vmax=1)
            axes[i, j].set_xlabel(f"Pred = {c[f'{key}_pred']:.0f}, Error = {c[f'{key}_error']:+.0f}",
                                  fontsize=11, labelpad=6)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    plt.subplots_adjust(left=0.03, right=0.98, top=0.96, bottom=0.1, wspace=0.05, hspace=0.2)
    cbar = fig.colorbar(im, cax=fig.add_axes([0.42, 0.045, 0.50, 0.015]), orientation="horizontal")
    cbar.ax.tick_params(labelsize=8, length=2)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {save_path}")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = select_representative_seed(args.evaluation_dir, args.output_dir)
    print(f"Representative seed: {seed}")
    dfs = load_predictions(args.evaluation_dir)
    cases = select_cases(dfs)

    mdir = seed_model_dir(args.model_dir, seed)
    cnn, cnn_mean, cnn_std = load_model(ResNet18RiskRegressor, mdir / "resnet18_layout_risk.pt", device)
    vit, vit_mean, vit_std = load_model(ViTTinyRiskRegressor, mdir / "vit_tiny_layout_risk.pt", device)

    results, rows = [], []
    for c in cases:
        x, raw = load_and_preprocess(args.data_dir / "layout_images", c["layout_image_path"], device)
        with torch.no_grad():
            c["cnn_pred"] = cnn(x).item() * cnn_std + cnn_mean
            c["vit_pred"] = vit(x).item() * vit_std + vit_mean
        c["cnn_error"] = c["cnn_pred"] - c["true_value"]
        c["vit_error"] = c["vit_pred"] - c["true_value"]
        ref = dfs[seed].loc[c["sample_index"]]                       # must match the evaluation output
        assert abs(c["cnn_pred"] - ref["cnn_pred"]) < 0.5 and abs(c["vit_pred"] - ref["vit_pred"]) < 0.5
        results.append({"case": c, "raw": raw,
                        "cnn_cam": score_cam_resnet(cnn, x, device), "vit_cam": score_cam_vit(vit, x, device)})
        rows.append({"Case": c["label"], "Category": c["category"], "Extrapolated": c["extrap_vars"],
                     "Actual": int(c["true_value"]),
                     "ResNet-18 predicted": round(c["cnn_pred"]), "ResNet-18 error": round(c["cnn_error"]),
                     "ViT-Tiny predicted": round(c["vit_pred"]), "ViT-Tiny error": round(c["vit_error"])})
        print(rows[-1])

    pd.DataFrame(rows).to_csv(args.output_dir / "representative_cases.csv", index=False)
    plot_scorecam_3x3(results, args.output_dir / "fig_scorecam_3cases.png")


if __name__ == "__main__":
    main()
