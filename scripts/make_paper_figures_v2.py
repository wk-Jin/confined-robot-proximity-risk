"""Regenerate Figs. 6-10 and Tables 4, 6, A3 and A4 of the paper.

Inputs are the outputs of run_additional_evaluation.py and analyze_generalisation.py
and the CSV files of the additional evaluation dataset. Values in the tables are
means and sample standard deviations (ddof=1) across the three training seeds.

Example:
    python scripts/make_paper_figures_v2.py
"""

# %% [Cell 1] Paths
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--evaluation-dir", type=Path, default=Path("outputs/additional_evaluation"))
parser.add_argument("--data-dir", type=Path, default=Path("data/additional_evaluation_dataset"))
parser.add_argument("--analysis-dir", type=Path, default=Path("outputs/generalisation_analysis"))
parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_figures_v2"))
ARGS = parser.parse_args()

EVAL_DIR = ARGS.evaluation_dir
DATA_DIR = ARGS.data_dir
GA_DIR = ARGS.analysis_dir
OUTPUT_BASE = str(ARGS.output_dir)
os.makedirs(OUTPUT_BASE, exist_ok=True)
BASE_DIR = OUTPUT_BASE                   # used by the screening cell

TRAIN_SEEDS = [42, 43, 44]


# %% [Cell 2] Style (same as the original notebook)
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

mpl.rcParams['font.family'] = 'DejaVu Sans'

AXIS_LABEL_FONTSIZE = 9.5
TICK_LABELSIZE      = 8.5
LEGEND_FONTSIZE     = 8.5

LINE_WIDTH    = 1.7
MARKER_SIZE   = 4.4
BAND_ALPHA     = 0.13
BAND_ALPHA_CNN = 0.13
BAND_ALPHA_VIT = 0.13
BAND_EDGE_LINEWIDTH = 0.0

GRID_LINEWIDTH = 0.55
GRID_ALPHA     = 0.35

LINE_COLOR_CNN = '#1f77b4'
LINE_COLOR_VIT = '#d62728'
MUTED          = '#8c8c8c'
MODEL_MARKERS    = {'cnn': 'o', 'vit': 's'}
MODEL_LINESTYLES = {'cnn': '-', 'vit': '--'}
MODEL_HATCHES    = {'cnn': '', 'vit': '//'}

N_SEEDS = len(TRAIN_SEEDS)


def style_axis(ax):
    ax.tick_params(labelsize=TICK_LABELSIZE)
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA)


def save_fig(fig, path_no_ext):
    fig.savefig(path_no_ext + '.pdf')
    fig.savefig(path_no_ext + '.png', dpi=300, facecolor='white')
    print(f"Saved: {path_no_ext}.pdf / .png")


# %% [Cell 3] Load the revised results into the original data shapes
MODEL_KEY = {'resnet18': 'cnn', 'vit_tiny': 'vit'}
CAT_ORDER = ['C0', 'C1', 'C2', 'C3', 'C4']
CAT_MAP = {c: c for c in CAT_ORDER}       # x labels use CAT_MAP[c].split(' ')[0]


def to_wide(df, key_col):
    """long (train_seed, key, model, metrics...) -> wide (train_seed, key, cnn_mae, vit_mae, ...)"""
    df = df.copy()
    df['model'] = df['model'].map(MODEL_KEY)
    metrics = [c for c in df.columns if c not in ('train_seed', key_col, 'model', 'model_label', 'n')]
    wide = df.pivot_table(index=['train_seed', key_col], columns='model', values=metrics + ['n'])
    wide.columns = [f'{m}_{metric}' if metric != 'n' else f'n_{m}' for metric, m in wide.columns]
    wide = wide.reset_index()
    wide['n'] = wide['n_cnn']
    return wide


category_df = to_wide(pd.read_csv(EVAL_DIR / 'summary_category_by_seed.csv'), 'category')
risk_df     = to_wide(pd.read_csv(EVAL_DIR / 'summary_risk_level_by_seed.csv'), 'risk_level')

seed_dfs = {}
for seed in TRAIN_SEEDS:
    df = pd.read_csv(EVAL_DIR / f'seed_{seed}' / 'all_predictions.csv')
    seed_dfs[seed] = df.rename(columns={'resnet18_pred': 'cnn_pred', 'vit_tiny_pred': 'vit_pred'})

# Checks
assert sorted(category_df['category'].unique()) == CAT_ORDER, category_df['category'].unique()
assert all(len(d) == 1000 for d in seed_dfs.values())
print(category_df.groupby('category')[['cnn_mae', 'vit_mae']].mean().round(2))


# %% [Cell 4] Fig. 6  category_summary (MAE / R² / UPR)

x = np.arange(len(CAT_ORDER))
x_labels = [CAT_MAP[c].split(' ')[0] for c in CAT_ORDER]

fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.70), facecolor='white')
# top 0.95 -> 0.85 and legend anchor 1.08 -> 0.99: the original legend was saved outside the canvas
fig.subplots_adjust(left=0.07, right=0.99, top=0.85, bottom=0.20, wspace=0.42)

metric_specs = [('mae', 'MAE', axes[0], (0, None)),
                ('r2', 'R²', axes[1], None),
                ('upr', 'UPR (%)', axes[2], (0, 100))]
model_specs = [('cnn', 'ResNet-18', LINE_COLOR_CNN, BAND_ALPHA_CNN),
               ('vit', 'ViT-Tiny', LINE_COLOR_VIT, BAND_ALPHA_VIT)]

legend_handles, legend_labels = [], []
for metric_key, ylabel, ax, clip_range in metric_specs:
    for model_key, model_label, color, band_alpha in model_specs:
        stat = category_df.groupby('category')[f'{model_key}_{metric_key}'].agg(['mean', 'std']).loc[CAT_ORDER]
        mean_vals = stat['mean'].values
        std_vals  = stat['std'].fillna(0.0).values          # sample SD across seeds (ddof=1)
        lower = mean_vals - std_vals
        upper = mean_vals + std_vals
        if clip_range is not None:
            lo, hi = clip_range
            if lo is not None: lower = np.clip(lower, lo, None)
            if hi is not None: upper = np.clip(upper, None, hi)

        line, = ax.plot(x, mean_vals, marker=MODEL_MARKERS[model_key], linestyle=MODEL_LINESTYLES[model_key],
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, color=color, label=model_label, zorder=4)
        ax.fill_between(x, lower, upper, color=color, alpha=band_alpha, linewidth=BAND_EDGE_LINEWIDTH, zorder=2)

        if metric_key == 'mae':
            legend_handles.append(line)
            legend_labels.append(model_label)

    if metric_key == 'r2':
        ax.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
    elif metric_key == 'upr':
        ax.axhline(50, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.set_ylim(0, 100)

    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_xlabel('Evaluation category', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    style_axis(ax)

fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.99), ncol=2, frameon=False, fontsize=LEGEND_FONTSIZE)

save_fig(fig, os.path.join(OUTPUT_BASE, 'fig_category_summary_mae_r2_upr'))
plt.close(fig)


# %% [Cell 5] Fig. 7  trend_summary_combined (slope / Spearman)
x = np.arange(len(CAT_ORDER))
x_labels = [CAT_MAP[c].split(' ')[0] for c in CAT_ORDER]

fig, axes = plt.subplots(1, 2, figsize=(6.40, 3.15), facecolor='white')
fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.27, wspace=0.32)

metric_specs = [('slope', 'Slope', axes[0]),
                ('spearman', 'Spearman ρ', axes[1])]
model_specs = [('cnn', 'ResNet-18', LINE_COLOR_CNN, BAND_ALPHA_CNN),
               ('vit', 'ViT-Tiny', LINE_COLOR_VIT, BAND_ALPHA_VIT)]

legend_handles, legend_labels = [], []
for metric_key, ylabel, ax in metric_specs:
    for model_key, model_label, color, band_alpha in model_specs:
        stat = category_df.groupby('category')[f'{model_key}_{metric_key}'].agg(['mean', 'std']).loc[CAT_ORDER]
        mean_vals = stat['mean'].values
        std_vals  = stat['std'].fillna(0.0).values

        line, = ax.plot(x, mean_vals, marker=MODEL_MARKERS[model_key], linestyle=MODEL_LINESTYLES[model_key],
                        linewidth=LINE_WIDTH, markersize=MARKER_SIZE, color=color, label=model_label, zorder=4)
        ax.fill_between(x, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=band_alpha,
                        linewidth=BAND_EDGE_LINEWIDTH, zorder=2)

        if metric_key == 'slope':
            legend_handles.append(line)
            legend_labels.append(model_label)

    if metric_key == 'slope':
        ax.axhline(1.0, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    else:
        ax.set_ylim(-0.05, 1.05)

    ax.set_xticks(x); ax.set_xticklabels(x_labels)
    ax.set_xlabel('Evaluation category', fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    style_axis(ax)

fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False, fontsize=LEGEND_FONTSIZE,
           handlelength=2.2)

save_fig(fig, os.path.join(OUTPUT_BASE, 'fig_trend_summary_combined'))
plt.close(fig)


# %% [Cell 6] Fig. 8  screening_gain
SCREEN_AXIS_LABEL_FONTSIZE = 12
SCREEN_LEGEND_FONTSIZE = 10


def cumulative_gain_curve(df, pred_col, true_col="true_value", positive_quantile=0.75, n_points=50):
    n = len(df)
    threshold = df[true_col].quantile(positive_quantile)
    is_positive = (df[true_col] >= threshold).values
    n_positive = is_positive.sum()

    order = np.argsort(-df[pred_col].values)
    sorted_positive = is_positive[order]

    ks = np.linspace(0, 1, n_points)
    recalls = np.array([
        sorted_positive[: max(1, int(np.ceil(k * n)))].sum() / n_positive
        for k in ks
    ])
    return ks, recalls


def gain_curves_all_seeds(seed_dfs, pred_col, positive_quantile=0.75, n_points=50):
    ks_ref, all_recalls = None, []
    for seed, df in seed_dfs.items():
        ks_ref, recalls = cumulative_gain_curve(df, pred_col, positive_quantile=positive_quantile, n_points=n_points)
        all_recalls.append(recalls)
    all_recalls = np.array(all_recalls)
    return ks_ref, all_recalls.mean(axis=0), all_recalls.std(axis=0, ddof=1)   # sample SD (Section 3.9)


ks, cnn_mean, cnn_std = gain_curves_all_seeds(seed_dfs, "cnn_pred")
ks, vit_mean, vit_std = gain_curves_all_seeds(seed_dfs, "vit_pred")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
fig.subplots_adjust(left=0.13, right=0.96, top=0.95, bottom=0.13)

ax.plot([0, 1], [0, 1], color=MUTED, linestyle=":", linewidth=0.8, label="Random")
ax.plot(ks, cnn_mean, color=LINE_COLOR_CNN, linestyle=MODEL_LINESTYLES['cnn'],
        linewidth=LINE_WIDTH, label="ResNet-18")
ax.fill_between(ks, cnn_mean - cnn_std, cnn_mean + cnn_std,
                color=LINE_COLOR_CNN, alpha=BAND_ALPHA, linewidth=BAND_EDGE_LINEWIDTH)
ax.plot(ks, vit_mean, color=LINE_COLOR_VIT, linestyle=MODEL_LINESTYLES['vit'],
        linewidth=LINE_WIDTH, label="ViT-Tiny")
ax.fill_between(ks, vit_mean - vit_std, vit_mean + vit_std,
                color=LINE_COLOR_VIT, alpha=BAND_ALPHA, linewidth=BAND_EDGE_LINEWIDTH)

for spine in ax.spines.values():
    spine.set_linewidth(1.25)

ax.set_xlabel("Top-K screened fraction", fontsize=SCREEN_AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Recall of target samples", fontsize=SCREEN_AXIS_LABEL_FONTSIZE)
ax.legend(frameon=False, fontsize=SCREEN_LEGEND_FONTSIZE)
style_axis(ax)
ax.tick_params(axis='both', labelsize=10)
save_fig(fig, os.path.join(BASE_DIR, 'fig_screening_gain'))
plt.close(fig)

for k_val in [0.1, 0.2, 0.3, 0.4, 0.5]:
    idx = np.argmin(np.abs(ks - k_val))
    print(f"K={ks[idx]:.2f} | ResNet-18 recall={cnn_mean[idx]:.3f} | ViT-Tiny recall={vit_mean[idx]:.3f} | Random={ks[idx]:.3f}")


# %% [Cell 7] Fig. 9  risk_level_summary (MAE / ME)
RISK_ORDER = ['Low', 'Medium', 'High', 'Extreme']

risk_n = risk_df.groupby('risk_level')['n'].mean().round().astype(int).loc[RISK_ORDER]
risk_x_labels = [f'{lvl}\n(n={risk_n[lvl]})' for lvl in RISK_ORDER]

x = np.arange(len(RISK_ORDER))
bar_width = 0.32
YLIM_PAD_FRAC = 0.10

fig, axes = plt.subplots(1, 2, figsize=(6.40, 3.15), facecolor='white')
fig.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.27, wspace=0.32)

metric_specs = [('mae', 'MAE', axes[0], True),
                ('me', 'Mean error (ME)', axes[1], False)]
model_specs = [('cnn', 'ResNet-18', LINE_COLOR_CNN),
               ('vit', 'ViT-Tiny', LINE_COLOR_VIT)]

legend_handles, legend_labels = [], []
for metric_key, ylabel, ax, clip_nonneg in metric_specs:
    span_lo, span_hi = [], []
    for i, (model_key, model_label, color) in enumerate(model_specs):
        stat = risk_df.groupby('risk_level')[f'{model_key}_{metric_key}'].agg(['mean', 'std']).loc[RISK_ORDER]
        mean_vals = stat['mean'].values
        std_vals  = stat['std'].fillna(0.0).values

        if clip_nonneg:
            lower_err = np.minimum(std_vals, mean_vals)
            upper_err = std_vals
        else:
            lower_err = std_vals
            upper_err = std_vals

        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, mean_vals, width=bar_width, yerr=[lower_err, upper_err],
                      color=color, capsize=2.5, label=model_label,
                      hatch=MODEL_HATCHES[model_key], edgecolor='black', linewidth=0.6,
                      error_kw=dict(elinewidth=0.8, ecolor='black', capthick=0.8))
        if metric_key == 'mae':
            legend_handles.append(bars)
            legend_labels.append(model_label)

        span_lo.append(np.min(mean_vals - lower_err))
        span_hi.append(np.max(mean_vals + upper_err))

    data_lo, data_hi = min(span_lo), max(span_hi)
    pad = (data_hi - data_lo) * YLIM_PAD_FRAC
    if clip_nonneg:
        ax.set_ylim(0, data_hi + pad)
    else:
        ax.set_ylim(data_lo - pad, data_hi + pad)

    if metric_key == 'me':
        ax.axhline(0, color='black', linewidth=0.8)

    ax.set_xticks(x); ax.set_xticklabels(risk_x_labels, fontsize=TICK_LABELSIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE)
    style_axis(ax)

fig.legend(handles=legend_handles, labels=legend_labels, loc='upper center',
           bbox_to_anchor=(0.5, 0.97), ncol=2, frameon=False, fontsize=LEGEND_FONTSIZE)

save_fig(fig, os.path.join(OUTPUT_BASE, 'fig_risk_level_summary_mae_me'))
plt.close(fig)

# Interval limits used in Table A4 / Fig. 9 (25th, 75th, 90th percentiles of actual values)
y_true = seed_dfs[TRAIN_SEEDS[0]]['true_value']
print('Interval limits (25/75/90th percentiles):', np.percentile(y_true, [25, 75, 90]))


# %% [Cell 8] Fig. 10  errors beyond the original parameter ranges (evaluation samples)
# (a) mean signed error (predicted - actual) by fence position, ResNet-18 and ViT-Tiny
# (b) ResNet-18 mean signed error by pillar position; dots = training grid positions
# Means are over the evaluation samples sharing each position; bands = sample SD across seeds.
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

DEV_FENCE    = (0.350, 0.500)
DEV_PILLAR_X = (-0.45, -0.25)
DEV_PILLAR_Y = (-0.55, -0.35)
TRAIN_PILLAR_X = [-0.45, -0.35, -0.25]
TRAIN_PILLAR_Y = [-0.55, -0.45, -0.35]

base = seed_dfs[TRAIN_SEEDS[0]][['fence_x', 'pillar_x', 'pillar_y', 'true_value']].copy()
for col in ('fence_x', 'pillar_x', 'pillar_y'):
    base[col] = base[col].round(4)
for s in TRAIN_SEEDS:                       # same row order in every seed file
    assert np.allclose(seed_dfs[s]['true_value'].values, base['true_value'].values)
err_by_seed = {key: np.column_stack([seed_dfs[s][f'{key}_pred'].values - base['true_value'].values
                                     for s in TRAIN_SEEDS]) for key in ('cnn', 'vit')}


def mean_error(mask, key):
    """mean signed error per seed over the selected samples -> (mean, sample SD across seeds)"""
    per_seed = err_by_seed[key][mask].mean(axis=0)
    return per_seed.mean(), per_seed.std(ddof=1)


fig = plt.figure(figsize=(6.40, 3.15), facecolor='white')
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.50,
                      left=0.10, right=0.89, top=0.80, bottom=0.17)
ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])

# ---- (a) fence position
ax_a.axvspan(*DEV_FENCE, color='0.90', zorder=0, label='Original parameter range')
ax_a.axhline(0, color='black', linewidth=0.8, zorder=1)
fence_vals = np.sort(base['fence_x'].unique())
for key, label, color, band_alpha in (('cnn', 'ResNet-18', LINE_COLOR_CNN, BAND_ALPHA_CNN),
                                      ('vit', 'ViT-Tiny', LINE_COLOR_VIT, BAND_ALPHA_VIT)):
    stats = np.array([mean_error((base['fence_x'] == v).values, key) for v in fence_vals])
    ax_a.plot(fence_vals, stats[:, 0], marker=MODEL_MARKERS[key], linestyle=MODEL_LINESTYLES[key],
              linewidth=LINE_WIDTH, markersize=MARKER_SIZE, color=color, label=label, zorder=4)
    ax_a.fill_between(fence_vals, stats[:, 0] - stats[:, 1], stats[:, 0] + stats[:, 1],
                      color=color, alpha=band_alpha, linewidth=BAND_EDGE_LINEWIDTH, zorder=2)
ax_a.set_xticks(np.round(np.arange(0.35, 0.551, 0.05), 2))
ax_a.set_xlim(0.345, 0.565)
ax_a.set_xlabel('Fence position, fence_x (m)', fontsize=AXIS_LABEL_FONTSIZE)
ax_a.set_ylabel('Mean error (predicted − actual)', fontsize=AXIS_LABEL_FONTSIZE)
style_axis(ax_a)

# ---- (b) pillar position, ResNet-18
rows = []
for (px, py), g in base.groupby(['pillar_x', 'pillar_y']):
    me, _ = mean_error(((base['pillar_x'] == px) & (base['pillar_y'] == py)).values, 'cnn')
    rows.append((px, py, me, len(g)))
err = pd.DataFrame(rows, columns=['pillar_x', 'pillar_y', 'me', 'n'])

xs = np.round(np.arange(-0.55, -0.1999, 0.05), 2)
ys = np.round(np.arange(-0.60, -0.2999, 0.05), 2)
Z = np.full((len(ys), len(xs)), np.nan)
for r in err.itertuples():
    Z[np.argmin(np.abs(ys - r.pillar_y)), np.argmin(np.abs(xs - r.pillar_x))] = r.me
dx = dy = 0.05
x_edges = np.r_[xs - dx / 2, xs[-1] + dx / 2]
y_edges = np.r_[ys - dy / 2, ys[-1] + dy / 2]
lim = np.nanmax(np.abs(Z))
ax_b.set_facecolor('0.93')                      # positions without evaluation samples
mesh = ax_b.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(Z), cmap='RdBu_r',
                       norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim), shading='flat',
                       edgecolors='white', linewidth=0.6)
for r in err.itertuples():
    ax_b.text(r.pillar_x, r.pillar_y, f'{r.me:+.0f}', ha='center', va='center', fontsize=6.0,
              color='white' if abs(r.me) > 0.6 * lim else 'black')
gx, gy = np.meshgrid(TRAIN_PILLAR_X, TRAIN_PILLAR_Y)
ax_b.plot(gx.ravel(), gy.ravel(), linestyle='none', marker='o', markersize=2.6,
          color='black', zorder=5)
ax_b.add_patch(Rectangle((DEV_PILLAR_X[0] - dx / 2, DEV_PILLAR_Y[0] - dy / 2),
                         (DEV_PILLAR_X[1] - DEV_PILLAR_X[0]) + dx, (DEV_PILLAR_Y[1] - DEV_PILLAR_Y[0]) + dy,
                         fill=False, edgecolor='black', linewidth=1.2, linestyle='--'))
ax_b.set_xlim(x_edges[0], x_edges[-1]); ax_b.set_ylim(y_edges[0], y_edges[-1])
ax_b.set_xticks(xs[::2]); ax_b.set_yticks(ys[::2])
ax_b.set_xlabel('pillar_x (m)', fontsize=AXIS_LABEL_FONTSIZE)
ax_b.set_ylabel('pillar_y (m)', fontsize=AXIS_LABEL_FONTSIZE)
ax_b.tick_params(labelsize=TICK_LABELSIZE)
cax = make_axes_locatable(ax_b).append_axes('right', size='5%', pad=0.06)
cb = fig.colorbar(mesh, cax=cax)
cb.set_label('Mean error (predicted − actual)', fontsize=TICK_LABELSIZE)
cb.ax.tick_params(labelsize=TICK_LABELSIZE - 1)

handles, labels = ax_a.get_legend_handles_labels()
handles.append(Line2D([], [], linestyle='none', marker='o', markersize=2.6, color='black'))
labels.append('Training grid position')
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.995), ncol=4,
           frameon=False, fontsize=LEGEND_FONTSIZE, handlelength=2.2, columnspacing=1.2)

save_fig(fig, os.path.join(OUTPUT_BASE, 'fig_error_beyond_range'))
plt.close(fig)

fence_table = pd.DataFrame({'fence_x': fence_vals,
                            'n': [int((base['fence_x'] == v).sum()) for v in fence_vals],
                            **{f'{k}_me': [mean_error((base['fence_x'] == v).values, k)[0] for v in fence_vals]
                               for k in ('cnn', 'vit')}})
print(fence_table.round({'fence_x': 4, 'cnn_me': 1, 'vit_me': 1}).to_string(index=False))
print(err.sort_values(['pillar_y', 'pillar_x'], ascending=[False, True])
         .round({'pillar_x': 2, 'pillar_y': 2, 'me': 1}).to_string(index=False))


# %% [Cell 9] Paper tables (Table 4, Table 6, Table A3, Table A4)
# Formatting only: values come from the evaluation outputs (mean ± sample SD across the three seeds).
# Each table is printed and saved as CSV and LaTeX (booktabs) under <output-dir>/tables/.
TABLE_DIR = os.path.join(OUTPUT_BASE, 'tables')
os.makedirs(TABLE_DIR, exist_ok=True)

MINUS = '−'                            # typographic minus, as in the manuscript


def fmt(mean, sd, decimals, signed=False):
    """'15.19 ± 0.31' / '+3.39 ± 2.05' with a typographic minus"""
    m = f"{mean:+.{decimals}f}" if signed else f"{mean:.{decimals}f}"
    s = f"{sd:.{decimals}f}"
    return f"{m} ± {s}".replace('-', MINUS)


def seed_summary(df, keys):
    """mean and sample SD (ddof=1) across seeds for every metric, per model"""
    metrics = ['mae', 'rmse', 'me', 'r2', 'upr', 'slope', 'spearman']
    metrics = [m for m in metrics if m in df.columns]
    g = df.groupby(keys + ['model'])[metrics]
    return g.mean().join(g.std(ddof=1), lsuffix='_mean', rsuffix='_sd').reset_index()


def save_table(table, name, caption):
    table.to_csv(os.path.join(TABLE_DIR, f'{name}.csv'), index=False)
    latex = table.to_latex(index=False, escape=False, column_format='l' + 'c' * (table.shape[1] - 1))
    for a, b in (('%', '\\%'), ('±', '$\\pm$'), (MINUS, '$-$'), ('R²', '$R^2$'),
                 ('≤', '$\\le$'), ('≥', '$\\ge$'), ('> ', '$>$ '), ('–', '--')):
        latex = latex.replace(a, b)
    with open(os.path.join(TABLE_DIR, f'{name}.tex'), 'w', encoding='utf-8') as f:
        f.write(f'% {caption}\n' + latex)
    print(f'\n=== {caption} ===')
    print(table.to_string(index=False))
    print(f'Saved: {TABLE_DIR}/{name}.csv / .tex')


MODEL_LABEL = {'resnet18': 'ResNet-18', 'vit_tiny': 'ViT-Tiny'}

# ---------------- Table 4: categories, sample composition and clean-success rates
CATEGORY_DESCRIPTION = {
    'C0': 'All 4 groups use previously unused intermediate values within the original ranges',
    'C1': '1 extrapolated group + 3 intermediate within-range groups',
    'C2': '2 extrapolated groups + 2 intermediate within-range groups',
    'C3': '3 extrapolated groups + 1 intermediate within-range group',
    'C4': 'All 4 parameter groups in extrapolation condition',
}
gen = pd.read_csv(DATA_DIR / 'generation_summary.csv').set_index('category').loc[CAT_ORDER]
table4 = pd.DataFrame({
    'Category': CAT_ORDER,
    'Condition description': [CATEGORY_DESCRIPTION[c] for c in CAT_ORDER],
    'Simulated (n)': gen['n_simulated'].astype(int).values,
    'Clean-success rate (%)': (100 * gen['clean_rate']).round(1).values,
    'Samples (N)': gen['n_clean'].astype(int).values,
})
table4.loc[len(table4)] = ['All', '', int(gen['n_simulated'].sum()),
                           round(100 * gen['n_clean'].sum() / gen['n_simulated'].sum(), 1),
                           int(gen['n_clean'].sum())]
save_table(table4, 'table4_categories', 'Table 4: Definitions and sample composition of the C0-C4 categories')

fence_clean_path = GA_DIR / 'fence_clean_rate.csv'
if fence_clean_path.exists():
    fc = pd.read_csv(fence_clean_path)
    print('\nClean-success rate by fence position (for Section 3.8 text):')
    print(fc.assign(clean_rate=(100 * fc['clean_rate']).round(1)).to_string(index=False))

# ---------------- Table 6: overall performance on the 1,000 samples
ov = seed_summary(pd.read_csv(EVAL_DIR / 'summary_overall_by_seed.csv'), [])
table6 = pd.DataFrame({
    'Model': ov['model'].map(MODEL_LABEL),
    'MAE': [fmt(r.mae_mean, r.mae_sd, 2) for r in ov.itertuples()],
    'RMSE': [fmt(r.rmse_mean, r.rmse_sd, 2) for r in ov.itertuples()],
    'ME': [fmt(r.me_mean, r.me_sd, 2, signed=True) for r in ov.itertuples()],
    'UPR (%)': [fmt(r.upr_mean, r.upr_sd, 1) for r in ov.itertuples()],
    'R²': [fmt(r.r2_mean, r.r2_sd, 4) for r in ov.itertuples()],
})
save_table(table6, 'table6_overall', 'Table 6: Model performance on the complete additional evaluation dataset')

# ---------------- Table A3: category-wise performance
cat = seed_summary(pd.read_csv(EVAL_DIR / 'summary_category_by_seed.csv'), ['category'])
rows = []
for c in CAT_ORDER:
    row = {'Category': c}
    for model in ('resnet18', 'vit_tiny'):
        r = cat[(cat['category'] == c) & (cat['model'] == model)].iloc[0]
        tag = MODEL_LABEL[model]
        row[f'{tag} MAE'] = fmt(r.mae_mean, r.mae_sd, 2)
        row[f'{tag} R²'] = fmt(r.r2_mean, r.r2_sd, 4)
        row[f'{tag} UPR (%)'] = fmt(r.upr_mean, r.upr_sd, 1)
    rows.append(row)
tableA3 = pd.DataFrame(rows)
save_table(tableA3, 'tableA3_category', 'Table A3: Model performance across the C0-C4 evaluation categories')

# ---------------- Table A4: indicator-value intervals
# Same rule as run_additional_evaluation.assign_risk_levels: Low <= q25 < Medium <= q75 < High <= q90 < Extreme
y_true = seed_dfs[TRAIN_SEEDS[0]]['true_value']
q25, q75, q90 = np.percentile(y_true, [25, 75, 90])
# invaded_grid_count is an integer, so the limits are written as non-overlapping integer ranges
interval_range = {
    'Low':     f'≤ {q25:.0f}',
    'Medium':  f'{q25 + 1:.0f}–{q75:.0f}',
    'High':    f'{q75 + 1:.0f}–{q90:.0f}',
    'Extreme': f'≥ {q90 + 1:.0f}',
}
risk = seed_summary(pd.read_csv(EVAL_DIR / 'summary_risk_level_by_seed.csv'), ['risk_level'])
risk_n = pd.read_csv(EVAL_DIR / 'summary_risk_level_by_seed.csv').groupby('risk_level')['n'].first()
rows = []
for lvl in RISK_ORDER:
    row = {'Indicator-value interval': f'{lvl} ({interval_range[lvl]}; n = {int(risk_n[lvl])})'}
    for model in ('resnet18', 'vit_tiny'):
        r = risk[(risk['risk_level'] == lvl) & (risk['model'] == model)].iloc[0]
        tag = MODEL_LABEL[model]
        row[f'{tag} MAE'] = fmt(r.mae_mean, r.mae_sd, 2)
        row[f'{tag} ME'] = fmt(r.me_mean, r.me_sd, 2, signed=True)
        row[f'{tag} UPR (%)'] = fmt(r.upr_mean, r.upr_sd, 1)
    rows.append(row)
tableA4 = pd.DataFrame(rows)
save_table(tableA4, 'tableA4_intervals', 'Table A4: Prediction errors and underprediction rates by indicator-value interval')

# ---------------- Values quoted in the text (Section 4.3.1)
base_path = GA_DIR / 'c0_grid_interpolation.csv'
if base_path.exists():
    b = pd.read_csv(base_path)
    print('\nC0 versus simulator grid-interpolation baseline (Section 4.3.1 text):')
    print(b[['method', 'n', 'mae', 'mae_seed_sd', 'share_better_than_baseline']].round(2).to_string(index=False))
