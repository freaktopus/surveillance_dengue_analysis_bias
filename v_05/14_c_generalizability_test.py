import argparse
import json
import os
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.special import gammaln
from scipy.optimize import minimize_scalar

warnings.filterwarnings('ignore')
torch.set_num_threads(4)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(_SCRIPT_DIR, 'outputs')

DEFAULT_V01 = os.path.join(BASE_DIR, 'panel_weekly_kaski_v_01.csv')
DEFAULT_V02 = os.path.join(BASE_DIR, 'panel_weekly_kaski_v_02.csv')
DEFAULT_RETRAIN_DIR = os.path.join(BASE_DIR, 'nbnll_13_retrain_cluster2')

DEFAULT_START = pd.Timestamp('2024-01-01')   # under-reporting begins here
DEFAULT_END = pd.Timestamp('2024-10-25')     # last data week = w/c 2024-10-21
DEFAULT_PLOT_START = pd.Timestamp('2023-01-01')

TRAIN_FRAC = 0.70          # mirrors 13_retrain.py / 06_model_training.py
EXCLUDED_DISTRICTS = ['KASKI']
CLUSTER = 2
CONDITION_FILES = {'B1': 'biased_B1.csv', 'B2': 'biased_B2.csv',
                   'B3': 'biased_B3.csv', 'combined': 'biased_combined.csv'}


# ---- Models (identical to 13_retrain.py / 06_model_training.py) ----
class VanillaRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


MODEL_CLASSES = {'RNN': VanillaRNN, 'LSTM': LSTMModel}


# ---- NB-NLL scoring (identical to common.py / 15_error_metric.py) -----------
def nb_logpmf(y, mu, alpha):
    """Gamma-function extension of the NB pmf (size r = 1/alpha, mean mu)."""
    r = 1.0 / max(alpha, 1e-10)
    p0 = r / (r + mu)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(p0) + y * np.log1p(-p0))


def nb_dispersion_mom(series, floor=1e-3):
    """Method-of-moments NB dispersion alpha (var = mu + alpha * mu^2)."""
    s = np.asarray(series, dtype=float)
    mu = s.mean()
    var = s.var(ddof=1) if len(s) > 1 else 0.0
    a = (var - mu) / max(mu ** 2, 1e-12)
    return float(max(a, floor))


def nb_nll_fixed(true, pred, alpha):
    """Mean NB-NLL with a FIXED (shared) alpha — apples-to-apples mode."""
    t_ = np.maximum(np.asarray(true, dtype=float), 0.0)
    m_ = np.maximum(np.asarray(pred, dtype=float), 1e-8)
    return -float(np.mean(nb_logpmf(t_, m_, alpha)))


def nb_nll_profile(true, pred):
    """Mean NB-NLL with per-series bounded-MLE alpha (15's 'profile' mode).
    Returns (nll, alpha) or (nan, nan) when the pair is too short."""
    t_ = np.maximum(np.asarray(true, dtype=float), 0.0)
    m_ = np.maximum(np.asarray(pred, dtype=float), 1e-8)
    if len(t_) < 2:
        return np.nan, np.nan
    res = minimize_scalar(
        lambda la: -float(np.mean(nb_logpmf(t_, m_, np.exp(la)))),
        bounds=(np.log(1e-5), np.log(1e4)), method='bounded')
    alpha = float(np.exp(res.x))
    return -float(np.mean(nb_logpmf(t_, m_, alpha))), alpha


# ---- Data loading / validation ----
def load_eval_series(path, label):
    """Load a single-district weekly panel; validate structure. Returns (district, weeks, values)."""
    df = pd.read_csv(path)
    need = {'district', 'iso_year', 'iso_week', 'weekly_cases', 'week_start'}
    if not need.issubset(df.columns):
        raise SystemExit(f"{label}: missing columns {need - set(df.columns)}")
    districts = df['district'].unique()
    if len(districts) != 1:
        raise SystemExit(f"{label}: expected exactly one district, got {list(districts)}")
    df = df.copy()
    df['week_start'] = pd.to_datetime(df['week_start'])
    df = df.sort_values('week_start').reset_index(drop=True)
    gaps = df['week_start'].diff().dropna()
    if not (gaps == pd.Timedelta(days=7)).all():
        raise SystemExit(f"{label}: weeks are not contiguous weekly Mondays.")
    if df['weekly_cases'].isna().any():
        raise SystemExit(f"{label}: NaN weekly_cases present.")
    print(f"  {label}: district={districts[0]}, n_weeks={len(df)}, "
          f"{df['week_start'].iloc[0].date()} .. {df['week_start'].iloc[-1].date()}")
    return districts[0], df['week_start'].tolist(), df['weekly_cases'].astype(float).values


# ---- Naive-forecast denominators (MASE / RMSSE) ----
def naive_denoms(history, m_season=52):
    """Naive-forecast error scales on a 1-D history (original count scale).

    m=1  : d_t = y_t - y_{t-1}
    m=52 : d_t = y_t - y_{t-52}   (annual seasonality; NaN if history too short)
    Returns {'m1_mae','m1_rmse','m52_mae','m52_rmse'}.
    """
    d1 = np.diff(np.asarray(history, dtype=float))
    out = {'m1_mae': float(np.mean(np.abs(d1))),
           'm1_rmse': float(np.sqrt(np.mean(d1 ** 2)))}
    if len(history) > m_season:
        d52 = np.asarray(history[m_season:], dtype=float) - \
              np.asarray(history[:-m_season], dtype=float)
        out['m52_mae'] = float(np.mean(np.abs(d52)))
        out['m52_rmse'] = float(np.sqrt(np.mean(d52 ** 2)))
    else:
        out['m52_mae'] = np.nan
        out['m52_rmse'] = np.nan
    return out


# ---- MC training-split reconstruction (for the MC-scale denominators) ----
_MC_SPLIT_CACHE = {}


def mc_train_split(condition, draw, retrain_manifest):
    """Rebuild the cluster-2 aggregate training split (weeks <= cutoff, first
    TRAIN_FRAC) that a given checkpoint was trained on -- the exact series its
    scaler was fitted on. Returns 1-D np.array of weekly counts."""
    key = (condition, int(draw))
    if key in _MC_SPLIT_CACHE:
        return _MC_SPLIT_CACHE[key]

    train_end = pd.Timestamp(retrain_manifest['train_window']['end'])
    if condition == 'clean':
        clusters_df = pd.read_csv(os.path.join(BASE_DIR, 'district_clusters.csv'))
        weekly = pd.read_csv(os.path.join(BASE_DIR, 'panel_weekly.csv'))
        clusters_df = clusters_df[~clusters_df['district'].isin(EXCLUDED_DISTRICTS)]
        weekly = weekly[~weekly['district'].isin(EXCLUDED_DISTRICTS)].copy()
        weekly['week_start'] = pd.to_datetime(weekly['week_start'])
        weekly = weekly.merge(clusters_df[['district', 'cluster']], on='district')
        agg = (weekly[weekly['cluster'] == CLUSTER]
               .groupby('week_start')['weekly_cases'].sum().reset_index()
               .sort_values('week_start'))
        train_window = agg[agg['week_start'] <= train_end]['weekly_cases'].values
    else:
        fname = CONDITION_FILES[condition]
        df = pd.read_csv(os.path.join(BASE_DIR, fname))
        if 'district' in df.columns:
            df = df[~df['district'].isin(EXCLUDED_DISTRICTS)]
        df['week_start'] = pd.to_datetime(df['week_start'])
        sub = df[(df['cluster'] == CLUSTER) & (df['draw'] == int(draw))]
        agg = (sub.groupby('week_start')['biased_cases'].sum().reset_index()
               .sort_values('week_start'))
        train_window = agg[agg['week_start'] <= train_end]['biased_cases'].values

    n_tr = int(len(train_window) * TRAIN_FRAC)
    split = np.asarray(train_window[:n_tr], dtype=float)
    _MC_SPLIT_CACHE[key] = split
    return split


_MC_ALPHA_CACHE = {}


def mc_clean_alpha(retrain_manifest):
    """Leakage-free MoM alpha from the cluster-2 CLEAN train window (weeks <=
    cutoff) — identical to 13_retrain.py's alpha_cluster (same window, same
    estimator), so nb_nll_mc is directly comparable with the retrain run."""
    key = 'clean'
    if key in _MC_ALPHA_CACHE:
        return _MC_ALPHA_CACHE[key]
    train_end = pd.Timestamp(retrain_manifest['train_window']['end'])
    clusters_df = pd.read_csv(os.path.join(BASE_DIR, 'district_clusters.csv'))
    weekly = pd.read_csv(os.path.join(BASE_DIR, 'panel_weekly.csv'))
    clusters_df = clusters_df[~clusters_df['district'].isin(EXCLUDED_DISTRICTS)]
    weekly = weekly[~weekly['district'].isin(EXCLUDED_DISTRICTS)].copy()
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    weekly = weekly.merge(clusters_df[['district', 'cluster']], on='district')
    agg = (weekly[weekly['cluster'] == CLUSTER]
           .groupby('week_start')['weekly_cases'].sum().reset_index()
           .sort_values('week_start'))
    train_window = agg[agg['week_start'] <= train_end]['weekly_cases'].values
    alpha = nb_dispersion_mom(train_window)
    _MC_ALPHA_CACHE[key] = alpha
    return alpha


# ---- Checkpoint handling ----
def select_checkpoints(manifest, args):
    """Filter manifest runs by conditions/models/seed/draw limits; return run dicts."""
    runs = manifest['runs']
    runs = [r for r in runs if r['condition'] in args.conditions and r['model'] in args.models]
    if args.seeds is not None:
        runs = [r for r in runs if r['seed'] < args.seeds]
    if args.max_draws is not None:
        for cond in args.conditions:
            if cond == 'clean':
                continue
            draws = sorted({r['draw'] for r in runs
                            if r['condition'] == cond and r['draw'] >= 0})
            keep = set(draws[:args.max_draws])
            runs = [r for r in runs if r['condition'] != cond
                    or r['draw'] < 0 or r['draw'] in keep]
    weight_dir = os.path.join(args.retrain_dir, 'weights_cluster2_retrain')
    out = []
    for r in runs:
        path = os.path.join(weight_dir, r['checkpoint'])
        if not os.path.exists(path):
            print(f"  WARNING: missing checkpoint {r['checkpoint']}, skipped")
            continue
        out.append({**r, 'path': path})
    return out


def forward_pass_series(ckpt, values, clip=True):
    """One-step-ahead walk-forward over the FULL series, DISPATCHING on the
    checkpoint's recorded transform (scaler['type']): 'zscore' -> mu/std
    pipeline, 'log1p' -> log1p/expm1. The transform is applied UNCHANGED.
    Returns predictions aligned to weeks[t_in:]."""
    arch = ckpt['arch']
    ModelCls = MODEL_CLASSES[arch['name']]
    model = ModelCls(input_size=arch['input_size'], hidden_size=arch['hidden'])
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    t_in = arch['t_in']
    sc = ckpt['scaler']
    ttype = sc.get('type', 'zscore')
    if ttype == 'log1p':
        norm, denorm = np.log1p, np.expm1
    elif ttype == 'zscore':
        mu, std = float(sc['mu']), float(sc['std'])
        norm = lambda x: (x - mu) / std
        denorm = lambda y: y * std + mu
    else:
        raise SystemExit(f"Unknown scaler type {ttype!r}; refusing to forward-pass "
                         "with a guessed transform (mismatched transforms silently "
                         "explode predictions).")
    z = norm(np.asarray(values, dtype=np.float32))
    X = np.array([z[i:i + t_in] for i in range(len(z) - t_in)], dtype=np.float32)
    with torch.no_grad():
        p = model(torch.tensor(X).unsqueeze(-1)).numpy()
    preds = denorm(p)
    if clip:
        preds = np.clip(preds, 0.0, None)
    return preds


# ---- Metrics ----
def _safe_div(a, b):
    return float(a / b) if (b is not None and np.isfinite(b) and b > 0) else np.nan


def eval_metrics(y_true, y_pred, denoms_local, denoms_mc,
                 alpha_local=None, alpha_mc=None, alpha_mode='mom'):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    obs_mean = max(float(np.mean(y_true)), 1e-8)
    corr = float(np.corrcoef(y_true, y_pred)[0, 1]) if y_true.std() > 0 else np.nan
    out = {
        'n_weeks': len(y_true),
        'rmse': rmse,
        'mae': mae,
        'bias': float(np.mean(err)),
        'corr': corr,
        'rel_rmse': rmse / obs_mean,           # RMSE / mean observed of THIS series
        'mase_m1': _safe_div(mae, denoms_local.get('m1_mae')),
        'rmsse_m1': _safe_div(rmse, denoms_local.get('m1_rmse')),
        'mase_m52': _safe_div(mae, denoms_local.get('m52_mae')),
        'rmsse_m52': _safe_div(rmse, denoms_local.get('m52_rmse')),
        'mase_m1_mc': _safe_div(mae, denoms_mc.get('m1_mae')),
        'rmsse_m1_mc': _safe_div(rmse, denoms_mc.get('m1_rmse')),
    }
    # ---- NB-NLL (in sync with 15_error_metric.py) ----
    if alpha_mode == 'profile':
        nll, a_prof = nb_nll_profile(y_true, y_pred)
        out['nb_nll'] = nll
        out['nb_nll_mc'] = nb_nll_fixed(y_true, y_pred, alpha_mc) \
            if alpha_mc is not None else np.nan
        out['alpha_profile'] = a_prof
        out['alpha_local'] = np.nan
    else:  # 'mom' — shared leakage-free alpha(s), apples-to-apples
        out['nb_nll'] = nb_nll_fixed(y_true, y_pred, alpha_local) \
            if alpha_local is not None else np.nan
        out['nb_nll_mc'] = nb_nll_fixed(y_true, y_pred, alpha_mc) \
            if alpha_mc is not None else np.nan
        out['alpha_profile'] = np.nan
        out['alpha_local'] = alpha_local
    return out


# ---- Figures ----
def _ensemble(pred_matrix, q=(5, 95)):
    med = np.median(pred_matrix, axis=0)
    lo = np.percentile(pred_matrix, q[0], axis=0)
    hi = np.percentile(pred_matrix, q[1], axis=0)
    return med, lo, hi


def fig_4series(ax, model, weeks, eval_weeks, ens_v01, ens_v02, obs_v01, obs_v02,
                start, plot_start, title_extra=''):
    """The requested four-series graph on one axes."""
    wk = np.array(weeks)
    sel = wk >= plot_start

    ax.plot(wk[sel], obs_v01[sel], color='tab:red', ls='--', lw=1.6,
            label='Observed v01 (under-reported)')
    ax.plot(wk[sel], obs_v02[sel], color='tab:blue', lw=1.8,
            label='Observed v02 (corrected)')

    wk_e = np.array(eval_weeks)
    for ens, color, lbl in ((ens_v01, 'tab:orange', 'Forecast on v01 (under-reported)'),
                            (ens_v02, 'tab:green', 'Forecast on v02 (corrected)')):
        med, lo, hi = ens
        ax.fill_between(wk_e, lo, hi, color=color, alpha=0.18, lw=0)
        ax.plot(wk_e, med, color=color, lw=2.0, label=lbl + ' (median, 5-95%)')

    ax.axvline(start, color='gray', ls=':', lw=1.2)
    ymax = max(np.nanmax(obs_v02[sel]), np.nanmax(obs_v01[sel]),
               np.nanmax(ens_v02[2]), np.nanmax(ens_v01[2]))
    ax.text(start, ymax * 1.02, ' under-reporting begins ', color='gray',
            fontsize=8, ha='center')
    ax.set_title(f'{model}: 1-week-ahead walk-forward forecasts vs observations'
                 f'{title_extra}', fontsize=11)
    ax.set_ylabel('weekly cases')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)
    fig = ax.figure
    fig.autofmt_xdate()
    fig.tight_layout()


def make_figures(outdir, weeks, obs_v01, obs_v02, eval_weeks, by_model, start, plot_start):
    fig_dir = os.path.join(outdir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # per-model 4-series figure
    for model, ens in by_model.items():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig_4series(ax, model, weeks, eval_weeks, ens['v01'], ens['v02'],
                    obs_v01, obs_v02, start, plot_start,
                    title_extra=f'  [{ens["n_ckpt"]} checkpoints]')
        fig.savefig(os.path.join(fig_dir, f'fig_4series_{model}.png'), dpi=150)
        plt.close(fig)

    # combined 2-panel figure
    fig, axes = plt.subplots(1, len(by_model), figsize=(7 * len(by_model), 5.2),
                             squeeze=False)
    for ax, (model, ens) in zip(axes[0], by_model.items()):
        fig_4series(ax, model, weeks, eval_weeks, ens['v01'], ens['v02'],
                    obs_v01, obs_v02, start, plot_start)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_4series_combined.png'), dpi=150)
    plt.close(fig)

    # error distributions (eval window only, per checkpoint)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for metric, ax in zip(['rmse', 'mase_m1', 'nb_nll'], axes):
        labels, data, colors = [], [], []
        for model, ens in by_model.items():
            for ser, color, lbl in (('v02v02', 'tab:green', 'pred(v02) vs v02'),
                                    ('v01v02', 'tab:red', 'pred(v01) vs v02')):
                vals = np.array(ens['per_ckpt'][ser][metric], dtype=float)
                vals = vals[np.isfinite(vals)]
                labels.append(f'{model}\n{lbl}')
                data.append(vals)
                colors.append(color)
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55)
        ax.set_title(f'{metric} distribution across checkpoints (vs corrected truth v02)')
        ax.grid(alpha=0.25, axis='y')
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_error_distributions.png'), dpi=150)
    plt.close(fig)

    # ratio histogram + MASE scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    for model, ens in by_model.items():
        ratio = np.array(ens['per_ckpt']['ratio_mase_m1'], dtype=float)
        ratio = ratio[np.isfinite(ratio)]
        ax.hist(ratio, bins=12, alpha=0.55, label=f'{model} (median {np.median(ratio):.2f})')
    ax.axvline(1.0, color='k', ls='--', lw=1)
    ax.set_xlabel('MASE pred(v01) / MASE pred(v02)   [>1 = under-reported inputs hurt]')
    ax.set_title('Per-checkpoint MASE ratio (both vs corrected truth v02)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis='y')

    ax = axes[1]
    for model, ens in by_model.items():
        x = np.array(ens['per_ckpt']['v02v02']['mase_m1'], dtype=float)
        y = np.array(ens['per_ckpt']['v01v02']['mase_m1'], dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[ok], y[ok], s=28, alpha=0.7, label=model)
    lims = [0, max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'k--', lw=1, label='y = x')
    ax.set_xlabel('MASE: forecast on corrected inputs (vs v02)')
    ax.set_ylabel('MASE: forecast on under-reported inputs (vs v02)')
    ax.set_title('MASE per checkpoint (above diagonal = under-reporting hurts)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, 'fig_ratio_mase.png'), dpi=150)
    plt.close(fig)


# ---- Markdown helper ----
def md_table(df, floatfmt='{:.3f}'):
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: floatfmt.format(v) if np.isfinite(v) else '--')
    lines = ['| ' + ' | '.join(map(str, d.columns)) + ' |',
             '|' + '|'.join(['---'] * len(d.columns)) + '|']
    for _, row in d.iterrows():
        lines.append('| ' + ' | '.join(map(str, row.values)) + ' |')
    return '\n'.join(lines)


def agg_stats(sub, col):
    v = sub[col].dropna().values
    if len(v) == 0:
        return {'mean': np.nan, 'sd': np.nan, 'ci95': np.nan, 'median': np.nan}
    return {'mean': float(np.mean(v)), 'sd': float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
            'ci95': float(1.96 * np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else np.nan,
            'median': float(np.median(v))}


# ---- Main ----
def main():
    ap = argparse.ArgumentParser(
        description='Generalizability test of 13_retrain checkpoints on the '
                    'under-reported (v01) vs corrected (v02) KASKI series.')
    ap.add_argument('--v01', default=DEFAULT_V01)
    ap.add_argument('--v02', default=DEFAULT_V02)
    ap.add_argument('--retrain-dir', default=DEFAULT_RETRAIN_DIR)
    ap.add_argument('--outdir', default=os.path.join(BASE_DIR, 'nbnll_14_generalizability'))
    ap.add_argument('--alpha-mode', choices=['mom', 'profile'], default='mom',
                    help="NB-NLL dispersion: 'mom' (leakage-free MoM on the shared "
                         "pre-window history, apples-to-apples; default) or "
                         "'profile' (per-scoring bounded MLE, as 15_error_metric.py)")
    ap.add_argument('--start', default=str(DEFAULT_START.date()),
                    help='first evaluated week (default 2024-01-01)')
    ap.add_argument('--end', default=str(DEFAULT_END.date()),
                    help='last evaluated date; final data week w/c 2024-10-21')
    ap.add_argument('--conditions', nargs='+', default=['clean'])
    ap.add_argument('--models', nargs='+', default=['RNN', 'LSTM'])
    ap.add_argument('--seeds', type=int, default=None, help='keep seed < N')
    ap.add_argument('--max-draws', type=int, default=None, help='limit biased draws')
    ap.add_argument('--plot-start', default=str(DEFAULT_PLOT_START.date()))
    ap.add_argument('--no-clip', action='store_true',
                    help='disable clipping of negative forecasts to 0')
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    plot_start = pd.Timestamp(args.plot_start)
    clip = not args.no_clip
    models = [m for m in args.models if m in MODEL_CLASSES]
    os.makedirs(args.outdir, exist_ok=True)

    print('14_c_generalizability_test: '
          f'conditions={args.conditions}, models={models}, clip={clip}, '
          f'alpha-mode={args.alpha_mode}')
    print(f'Evaluation window: weeks {start.date()} .. {end.date()} '
          '(last data week = w/c 2024-10-21)')

    # ---- 1. Load + validate both series ----
    print('Loading series:')
    dist_a, weeks, v01 = load_eval_series(args.v01, 'v01 under-reported')
    dist_b, weeks_b, v02 = load_eval_series(args.v02, 'v02 corrected')
    if dist_a != dist_b:
        raise SystemExit(f'District mismatch: {dist_a} vs {dist_b}')
    if weeks != weeks_b:
        raise SystemExit('The two files do not cover identical week sets.')
    weeks = pd.DatetimeIndex(weeks)

    pre_mask = weeks < start
    pre_identical = bool(np.allclose(np.array(v01)[pre_mask], np.array(v02)[pre_mask]))
    eval_mask = (weeks >= start) & (weeks <= end)
    eval_weeks = weeks[eval_mask]
    print(f'  eval weeks: {len(eval_weeks)} ({eval_weeks[0].date()} .. {eval_weeks[-1].date()}); '
          f'pre-window history: {int(pre_mask.sum())} weeks; '
          f'v01==v02 before window: {pre_identical}')

    t_in_default = 8
    first_pred_week = weeks[t_in_default]
    if eval_weeks[0] < first_pred_week:
        raise SystemExit('Not enough history before the evaluation window.')

    # ---- 2. Denominators + NB-NLL alphas ----
    denoms_v01 = naive_denoms(np.array(v01)[pre_mask])
    denoms_v02 = naive_denoms(np.array(v02)[pre_mask])
    print(f'  local naive scales  v01: {denoms_v01}')
    print(f'                      v02: {denoms_v02}')

    manifest_path = os.path.join(args.retrain_dir, 'manifest.json')
    with open(manifest_path) as f:
        retrain_manifest = json.load(f)

    # NB-NLL alphas (in sync with 15_error_metric.py):
    # LOCAL: MoM on the pre-window history (v01 == v02 there -> shared alpha,
    # apples-to-apples for both forecasts). MC-SCALE: MoM on the cluster-2
    # CLEAN train window <= cutoff (identical to 13_retrain.py).
    alpha_local = nb_dispersion_mom(np.array(v02)[pre_mask])
    alpha_mc = mc_clean_alpha(retrain_manifest)
    print(f'  NB-NLL alpha (local MoM, pre-window v02): {alpha_local:.4f}')
    print(f'  NB-NLL alpha (MC clean train, <= cutoff): {alpha_mc:.4f}')

    ckpts = select_checkpoints(retrain_manifest, args)
    if not ckpts:
        raise SystemExit('No checkpoints selected.')
    print(f'Checkpoints selected: {len(ckpts)} '
          f'({sorted({c["model"] for c in ckpts})} x {sorted({c["condition"] for c in ckpts})})')

    # ---- 3. Walk-forward per checkpoint ----
    per_ck_rows, ratio_rows, full_pred_rows = [], [], []
    by_model = {m: {'v01': None, 'v02': None, 'per_ckpt':
                    {'v02v02': {}, 'v01v02': {}, 'ratio_mase_m1': [], 'ratio_rmse': [],
                     'ratio_rel_rmse': [], 'nll_improvement_pct': []}, 'n_ckpt': None}
                for m in models}
    coll = {m: {'v01': [], 'v02': []} for m in models}
    eval_set = set(eval_weeks)
    types_seen = set()

    for i, run in enumerate(ckpts):
        ckpt = torch.load(run['path'], map_location='cpu', weights_only=False)
        types_seen.add(ckpt['scaler'].get('type', 'zscore'))
        t_in = ckpt['arch']['t_in']
        split_mc = mc_train_split(run['condition'], run['draw'], retrain_manifest)
        denoms_mc = naive_denoms(split_mc)

        p01_all = forward_pass_series(ckpt, v01, clip=clip)
        p02_all = forward_pass_series(ckpt, v02, clip=clip)
        p01 = pd.Series(p01_all, index=weeks[t_in:]).loc[eval_weeks].values
        p02 = pd.Series(p02_all, index=weeks[t_in:]).loc[eval_weeks].values
        y01 = np.array(v01)[eval_mask]
        y02 = np.array(v02)[eval_mask]

        # ---- Scoring: BOTH forecasts against the CORRECTED truth (v02) ----
        # 1) forecast built on corrected inputs      (pred_v02)  vs v02
        m_pred02 = eval_metrics(y02, p02, denoms_v01, denoms_mc,
                                alpha_local=alpha_local, alpha_mc=alpha_mc,
                                alpha_mode=args.alpha_mode)
        # 2) forecast built on under-reported inputs (pred_v01) vs SAME v02 truth
        m_pred01 = eval_metrics(y02, p01, denoms_v01, denoms_mc,
                                alpha_local=alpha_local, alpha_mc=alpha_mc,
                                alpha_mode=args.alpha_mode)
        # supplementary self-tracking: forecast-on-v01 vs v01 itself
        m_self = eval_metrics(y01, p01, denoms_v01, denoms_mc,
                              alpha_local=alpha_local, alpha_mc=alpha_mc,
                              alpha_mode=args.alpha_mode)

        base = {'model': run['model'], 'condition': run['condition'],
                'draw': run['draw'], 'seed': run['seed'],
                'checkpoint': run['checkpoint']}
        per_ck_rows.append({**base, 'scoring': 'predv02_vs_v02', **m_pred02})
        per_ck_rows.append({**base, 'scoring': 'predv01_vs_v02', **m_pred01})
        per_ck_rows.append({**base, 'scoring': 'predv01_vs_v01_selftrack', **m_self})
        ratio_rows.append({**base,
                           'rmse_predv02_vs_v02': m_pred02['rmse'],
                           'rmse_predv01_vs_v02': m_pred01['rmse'],
                           'ratio_rmse_predv01_over_predv02':
                               _safe_div(m_pred01['rmse'], m_pred02['rmse']),
                           'relrmse_predv02_vs_v02': m_pred02['rel_rmse'],
                           'relrmse_predv01_vs_v02': m_pred01['rel_rmse'],
                           'ratio_relrmse_predv01_over_predv02':
                               _safe_div(m_pred01['rel_rmse'], m_pred02['rel_rmse']),
                           'mase_m1_predv02_vs_v02': m_pred02['mase_m1'],
                           'mase_m1_predv01_vs_v02': m_pred01['mase_m1'],
                           'ratio_mase_m1_predv01_over_predv02':
                               _safe_div(m_pred01['mase_m1'], m_pred02['mase_m1']),
                           'rmsse_m1_predv02_vs_v02': m_pred02['rmsse_m1'],
                           'rmsse_m1_predv01_vs_v02': m_pred01['rmsse_m1'],
                           'ratio_rmsse_m1_predv01_over_predv02':
                               _safe_div(m_pred01['rmsse_m1'], m_pred02['rmsse_m1']),
                           'nll_predv02_vs_v02': m_pred02['nb_nll'],
                           'nll_predv01_vs_v02': m_pred01['nb_nll'],
                           'nll_improvement_pct_predv02_over_predv01':
                               ((m_pred01['nb_nll'] - m_pred02['nb_nll']) /
                                abs(m_pred01['nb_nll']) * 100.0
                                if (np.isfinite(m_pred01['nb_nll']) and
                                    np.isfinite(m_pred02['nb_nll']) and
                                    m_pred01['nb_nll'] != 0) else np.nan),
                           'nll_mc_predv02_vs_v02': m_pred02['nb_nll_mc'],
                           'nll_mc_predv01_vs_v02': m_pred01['nb_nll_mc'],
                           'predv01_worse_mase_m1':
                               bool(np.isfinite(m_pred01['mase_m1']) and
                                    np.isfinite(m_pred02['mase_m1']) and
                                    m_pred01['mase_m1'] > m_pred02['mase_m1']),
                           'predv01_worse_nll':
                               bool(np.isfinite(m_pred01['nb_nll']) and
                                    np.isfinite(m_pred02['nb_nll']) and
                                    m_pred01['nb_nll'] > m_pred02['nb_nll'])})

        # ---- per-point predictions: EVERY point of the walk-forward sequence ----
        for j, w in enumerate(weeks[t_in:]):
            full_pred_rows.append({'model': run['model'], 'condition': run['condition'],
                                   'draw': run['draw'], 'seed': run['seed'],
                                   'checkpoint': run['checkpoint'],
                                   'week_start': str(w.date()),
                                   'pred_on_v01_underreported': float(p01_all[j]),
                                   'pred_on_v02_corrected': float(p02_all[j]),
                                   'obs_v01_underreported': float(v01[t_in + j]),
                                   'obs_v02_corrected': float(v02[t_in + j]),
                                   'in_eval_window': bool(w in eval_set)})

        coll[run['model']]['v01'].append(p01)
        coll[run['model']]['v02'].append(p02)
        pc = by_model[run['model']]['per_ckpt']
        for ser, mm in (('v02v02', m_pred02), ('v01v02', m_pred01)):
            for k, v in mm.items():
                pc[ser].setdefault(k, []).append(v)
        pc['ratio_mase_m1'].append(_safe_div(m_pred01['mase_m1'], m_pred02['mase_m1']))
        pc['ratio_rmse'].append(_safe_div(m_pred01['rmse'], m_pred02['rmse']))
        pc['ratio_rel_rmse'].append(_safe_div(m_pred01['rel_rmse'], m_pred02['rel_rmse']))
        pc['nll_improvement_pct'].append(
            (m_pred01['nb_nll'] - m_pred02['nb_nll']) / abs(m_pred01['nb_nll']) * 100.0
            if (np.isfinite(m_pred01['nb_nll']) and np.isfinite(m_pred02['nb_nll'])
                and m_pred01['nb_nll'] != 0) else np.nan)

        if (i + 1) % 10 == 0 or i + 1 == len(ckpts):
            print(f'  [{i + 1}/{len(ckpts)}] {run["model"]} {run["condition"]} '
                  f'draw{run["draw"]} seed{run["seed"]}: '
                  f'RMSE pred(v02)vs v02={m_pred02["rmse"]:.1f} '
                  f'pred(v01)vs v02={m_pred01["rmse"]:.1f} | '
                  f'MASE {m_pred02["mase_m1"]:.2f} vs {m_pred01["mase_m1"]:.2f} | '
                  f'NLL {m_pred02["nb_nll"]:.2f} vs {m_pred01["nb_nll"]:.2f}')

    # ---- 4. Ensembles + figures ----
    obs_v01 = np.array(v01)
    obs_v02 = np.array(v02)
    for m in models:
        n = len(coll[m]['v01'])
        by_model[m]['n_ckpt'] = n
        by_model[m]['v01'] = _ensemble(np.array(coll[m]['v01']))
        by_model[m]['v02'] = _ensemble(np.array(coll[m]['v02']))
    make_figures(args.outdir, list(weeks), obs_v01, obs_v02, eval_weeks,
                 by_model, start, plot_start)

    # ---- 5. Tables ----
    tab_dir = os.path.join(args.outdir, 'tables')
    os.makedirs(tab_dir, exist_ok=True)
    per_ck = pd.DataFrame(per_ck_rows)
    ratios = pd.DataFrame(ratio_rows)
    per_ck.to_csv(os.path.join(tab_dir, 'per_checkpoint_metrics.csv'), index=False)
    ratios.to_csv(os.path.join(tab_dir, 'checkpoint_ratios.csv'), index=False)
    # EVERY point of EVERY checkpoint's walk-forward sequence
    pd.DataFrame(full_pred_rows).to_csv(
        os.path.join(tab_dir, 'predictions_full_sequence.csv'), index=False)

    metric_cols = ['rmse', 'mae', 'bias', 'corr', 'rel_rmse', 'mase_m1', 'rmsse_m1',
                   'mase_m52', 'rmsse_m52', 'mase_m1_mc', 'rmsse_m1_mc',
                   'nb_nll', 'nb_nll_mc', 'alpha_profile']
    sum_rows = []
    for model in models:
        for cond in sorted({r['condition'] for r in ckpts}):
            for scoring in ('predv02_vs_v02', 'predv01_vs_v02',
                            'predv01_vs_v01_selftrack'):
                sub = per_ck[(per_ck['model'] == model) & (per_ck['condition'] == cond) &
                             (per_ck['scoring'] == scoring)]
                if not len(sub):
                    continue
                row = {'model': model, 'condition': cond, 'scoring': scoring,
                       'n_checkpoints': len(sub)}
                for c in metric_cols:
                    st = agg_stats(sub, c)
                    row.update({f'{c}_mean': st['mean'], f'{c}_sd': st['sd'],
                                f'{c}_ci95': st['ci95'], f'{c}_median': st['median']})
                sum_rows.append(row)
    summary = pd.DataFrame(sum_rows)
    summary.to_csv(os.path.join(tab_dir, 'summary_by_model.csv'), index=False)

    wk_rows = []
    for m in models:
        med01, lo01, hi01 = by_model[m]['v01']
        med02, lo02, hi02 = by_model[m]['v02']
        for j, w in enumerate(eval_weeks):
            pos = weeks.get_loc(w)
            wk_rows.append({'model': m, 'week_start': str(w.date()),
                            'obs_v01': obs_v01[pos], 'obs_v02': obs_v02[pos],
                            'pred_v01_median': med01[j], 'pred_v01_q05': lo01[j],
                            'pred_v01_q95': hi01[j],
                            'pred_v02_median': med02[j], 'pred_v02_q05': lo02[j],
                            'pred_v02_q95': hi02[j]})
    pd.DataFrame(wk_rows).to_csv(os.path.join(tab_dir, 'weekly_ensemble.csv'), index=False)

    # ---- 6. Report ----
    headline = {}
    for m in models:
        sub = ratios[ratios['model'] == m]
        headline[m] = {
            'n': len(sub),
            'rmse_pred02_mean': sub['rmse_predv02_vs_v02'].mean(),
            'rmse_pred01_mean': sub['rmse_predv01_vs_v02'].mean(),
            'ratio_rmse_med': sub['ratio_rmse_predv01_over_predv02'].median(),
            'relrmse_pred02_mean': sub['relrmse_predv02_vs_v02'].mean(),
            'relrmse_pred01_mean': sub['relrmse_predv01_vs_v02'].mean(),
            'mase02_med': sub['mase_m1_predv02_vs_v02'].median(),
            'mase01_med': sub['mase_m1_predv01_vs_v02'].median(),
            'ratio_mase_med': sub['ratio_mase_m1_predv01_over_predv02'].median(),
            'pct_v01_worse': 100.0 * sub['predv01_worse_mase_m1'].mean(),
            'rmsse02_med': sub['rmsse_m1_predv02_vs_v02'].median(),
            'rmsse01_med': sub['rmsse_m1_predv01_vs_v02'].median(),
            'nll02_mean': sub['nll_predv02_vs_v02'].mean(),
            'nll01_mean': sub['nll_predv01_vs_v02'].mean(),
            'nll_improvement_med': sub['nll_improvement_pct_predv02_over_predv01'].median(),
            'pct_v01_worse_nll': 100.0 * sub['predv01_worse_nll'].mean(),
        }

    sum_cols = ['model', 'scoring', 'n_checkpoints', 'rmse_mean', 'rmse_sd',
                'rel_rmse_mean', 'mase_m1_mean', 'mase_m1_median',
                'rmsse_m1_mean', 'mase_m52_mean', 'nb_nll_mean', 'nb_nll_mc_mean']
    sum_show = summary[[c for c in sum_cols if c in summary.columns]]
    ratio_show = ratios[['model', 'draw', 'seed', 'rmse_predv02_vs_v02',
                         'rmse_predv01_vs_v02', 'ratio_rmse_predv01_over_predv02',
                         'mase_m1_predv02_vs_v02', 'mase_m1_predv01_vs_v02',
                         'ratio_mase_m1_predv01_over_predv02',
                         'nll_predv02_vs_v02', 'nll_predv01_vs_v02',
                         'nll_improvement_pct_predv02_over_predv01', 'predv01_worse_nll']]

    v02_level_ratio = float(np.mean(np.array(v02)[eval_mask]) /
                            max(np.mean(np.array(v01)[eval_mask]), 1e-8))
    lines = [
        '# Generalizability test (NB-NLL): 13_retrain checkpoints on KASKI v01 (under-reported) vs v02 (corrected)',
        '',
        f"*Generated by `14_c_generalizability_test.py` on {pd.Timestamp.today().date()}.*",
        '',
        '## 1. Objective',
        '',
        'Assess whether the RNN/LSTM forecasters retrained in the NB-NLL generation of '
        '`13_retrain.py` '
        '(cluster-2 MC data, hard cutoff 2024-10-21) generalize to two versions of the '
        f'KASKI weekly series: the **under-reported** panel (`panel_weekly_kaski_v_01.csv`) '
        f'and the **corrected** panel (`panel_weekly_kaski_v_02.csv`). Every scoring is '
        'evaluated with the new error metric (NB-NLL, in sync with `15_error_metric.py`) '
        'in addition to RMSE/MASE/RMSSE.',
        '',
        '## 2. Data',
        '',
        f'- District: **{dist_a}**; {len(weeks)} contiguous ISO weeks, '
        f'{weeks[0].date()} -> {weeks[-1].date()} (identical week grids in both files).',
        f'- The two series are identical before {start.date()}: **{pre_identical}** '
        '(verified at runtime).',
        f'- Evaluation window: **{len(eval_weeks)} weeks** ({eval_weeks[0].date()} .. '
        f'{eval_weeks[-1].date()}), i.e. every week whose ISO week contains dates up to '
        f'{end.date()}. The under-reporting discrepancy exists throughout this window.',
        f'- 2024 level gap: mean weekly cases v01 = '
        f'{np.mean(np.array(v01)[eval_mask]):.1f} vs v02 = '
        f'{np.mean(np.array(v02)[eval_mask]):.1f} '
        f'(v02/v01 level ratio = {v02_level_ratio:.2f}) -- this level difference is the '
        'scale confound that the scaled metrics below remove.',
        '',
        '## 3. Method',
        '',
        '- **Walk-forward 1-week-ahead**: for every evaluated week the model receives the '
        f'{8} preceding OBSERVED weeks of the *same* series (sliding T_IN=8 windows); '
        'predictions are never fed back.',
        '- **Common ground truth**: BOTH walk-forward forecasts are scored against the '
        'CORRECTED series (v02): (1) forecast built on corrected inputs (pred_v02) vs v02; '
        '(2) forecast built on under-reported inputs (pred_v01) vs v02. Same truth for both '
        '-> raw RMSE / relative-RMSE comparisons are apples-to-apples with no level confound. '
        'A supplementary self-tracking scoring (pred_v01 vs v01 itself) is recorded in the '
        'per-checkpoint table.',
        f'- **Frozen transforms**: each checkpoint\'s own transform, recorded as '
        f'scaler["type"] = {sorted(types_seen)} and dispatched at load time '
        '(zscore -> mu/std pipeline; log1p -> log1p/expm1), is applied UNCHANGED '
        'to both series. No transform is refit on v01 or v02.',
        f'- **Checkpoints used**: {len(ckpts)} '
        f'(conditions {sorted(set(r["condition"] for r in ckpts))}, '
        f'models {sorted(set(r["model"] for r in ckpts))}); '
        'per-model figures aggregate checkpoints by median + 5-95% band.',
        f'- **Negative-forecast clipping**: {"ON" if clip else "OFF"} (case counts cannot be negative).',
        '',
        '### NB-NLL scoring (in sync with 15_error_metric.py)',
        '',
        f'- NB-NLL(y | mu, alpha) with var = mu + alpha*mu^2; mu = walk-forward prediction, '
        'y = the scored series.',
        f'- **Alpha mode: {args.alpha_mode}**.',
        f'- LOCAL alpha (MoM on the shared pre-window history, where v01 == v02): '
        f'**{alpha_local:.4f}** — both forecasts share this alpha, so the '
        'pred(v01)-vs-pred(v02) NLL comparison is apples-to-apples.',
        f'- MC-SCALE alpha (MoM on the cluster-2 CLEAN train window <= cutoff, identical '
        f'to 13_retrain.py): **{alpha_mc:.4f}** — recorded as nb_nll_mc for direct '
        'comparison with the retrain-run NLLs.',
        '- In `profile` mode alpha is fitted per scoring by bounded MLE on the scored '
        'series (recorded in alpha_profile); the NLL comparison is then no longer '
        'strictly apples-to-apples.',
        '',
        '### MASE / RMSSE denominators',
        '',
        f'- **LOCAL (primary)**: naive-forecast errors on the series\' own pre-window history '
        f'({int(pre_mask.sum())} weeks before {start.date()}), m=1 and seasonal m=52. '
        'Because v01 and v02 are identical there, BOTH series share the *same* denominator '
        '(`m1_mae` = %.3f, `m1_rmse` = %.3f) -- the v02-vs-v01 comparison is therefore '
        'scale-free by construction.' % (denoms_v01['m1_mae'], denoms_v01['m1_rmse']),
        f'- **MC-SCALE (secondary)**: naive errors on each checkpoint\'s own MC training split '
        f'(first {TRAIN_FRAC:.0%} of the cluster-2 weeks <= cutoff). This sits on the '
        'cluster-aggregate scale (much larger than a single district), so absolute values are '
        'small; it is reported for cross-scale comparability only.',
        '',
        'MASE = MAE / mean|naive error|; RMSSE = RMSE / sqrt(mean naive error^2). '
        'MASE < 1 beats the naive forecast.',
        '',
        '## 4. Headline results (per model, across checkpoints)',
        '',
    ]
    for m, h in headline.items():
        lines += [
            f'**{m}** ({h["n"]} checkpoints, both scored against corrected truth v02):',
            '',
            f'- RMSE pred(v02) vs v02: **{h["rmse_pred02_mean"]:.1f}**; '
            f'RMSE pred(v01) vs v02: **{h["rmse_pred01_mean"]:.1f}**.',
            f'- Per-checkpoint RMSE ratio pred(v01)/pred(v02) median: '
            f'**{h["ratio_rmse_med"]:.2f}** (>1 = under-reported inputs degrade accuracy '
            'against the corrected truth).',
            f'- Relative RMSE (RMSE / mean of v02 = {np.mean(np.array(v02)[eval_mask]):.1f}, '
            f'common to both): pred(v02) {h["relrmse_pred02_mean"]:.2f} vs '
            f'pred(v01) {h["relrmse_pred01_mean"]:.2f}.',
            f'- **MASE(m=1, shared local denominator): pred(v02) median {h["mase02_med"]:.2f} '
            f'vs pred(v01) median {h["mase01_med"]:.2f}**; per-checkpoint ratio median '
            f'**{h["ratio_mase_med"]:.2f}**; the under-reported-input forecast is worse in '
            f'**{h["pct_v01_worse"]:.0f}%** of checkpoints.',
            f'- RMSSE(m=1) medians: pred(v02) {h["rmsse02_med"]:.2f}, '
            f'pred(v01) {h["rmsse01_med"]:.2f}.',
            f'- **NB-NLL (shared local alpha = {alpha_local:.4f}): pred(v02) mean '
            f'{h["nll02_mean"]:.2f} vs pred(v01) mean {h["nll01_mean"]:.2f}**; '
            f'per-checkpoint NLL improvement of corrected inputs (15_error_metric.py '
            f'formula) median **{h["nll_improvement_med"]:.2f}%** '
            '(positive = corrected-input forecast has the lower NLL); the '
            f'under-reported-input forecast has the worse NLL in '
            f'**{h["pct_v01_worse_nll"]:.0f}%** of checkpoints.',
            '',
        ]
    lines += [
        '## 5. Summary table (mean across checkpoints)',
        '',
        md_table(sum_show),
        '',
        '## 6. Per-checkpoint ratios (first 20 rows)',
        '',
        md_table(ratio_show.head(20), floatfmt='{:.2f}'),
        '',
        '## 7. Figures',
        '',
        '- `figures/fig_4series_RNN.png`, `figures/fig_4series_LSTM.png` -- the requested '
        'four-series graph: observed v01/v02 + walk-forward forecast of each (median + 5-95% band).',
        '- `figures/fig_4series_combined.png` -- both models side by side.',
        '- `figures/fig_error_distributions.png` -- RMSE, MASE and NB-NLL distributions: '
        'pred(v02) vs pred(v01), both scored against the corrected truth v02.',
        '- `figures/fig_ratio_mase.png` -- MASE ratio histogram and per-checkpoint scatter.',
        '',
        '## 8. Interpretation guide & caveats',
        '',
        '- Because both forecasts are scored against the SAME corrected truth (v02) and '
        'share the SAME MASE/RMSSE denominator, every ratio is apples-to-apples: '
        '**ratio > 1 means forecasting from under-reported inputs is less accurate than '
        'forecasting from corrected inputs** when judged against the true (corrected) '
        'incidence. There is no level confound in this comparison.',
        '- Relative RMSE divides by the mean of v02 in the window (common to both '
        'forecasts), so its ratio equals the raw RMSE ratio; it expresses the error as a '
        'fraction of average corrected incidence.',
        '- MASE/RMSSE > 1 additionally signal that neither forecast beats the naive '
        'one-week-ahead forecast on the shared pre-window scale.',
        '- NB-NLL is a proper scoring rule for count data: lower is better, and with the '
        'SHARED local alpha the pred(v01)-vs-pred(v02) NLL comparison is apples-to-apples '
        '(same truth, same dispersion). NLL improvement (15_error_metric.py formula: '
        '(NLL_v01 - NLL_v02)/|NLL_v01| * 100) > 0 means forecasting from corrected inputs '
        'achieves the lower (better) NLL. '
        'nb_nll_mc rescores with the retrain-run alpha for direct comparison with '
        '`nbnll_13_retrain_cluster2/retrain_results.csv`.',
        '- All forecasts are one-step-ahead on observed history; nothing beyond '
        f'{weeks[-1].date()} is predicted (recursive extrapolation is out of scope here).',
        '- Models were trained on the cluster-2 aggregate (mean ~500/wk); KASKI is a single '
        'district (~30-165/wk in the eval window), so all inputs are out-of-magnitude '
        'relative to training -- generalizability in the strict sense, not in-domain accuracy.',
        '- Checkpoint transforms were frozen as required (dispatched on the recorded '
        'scaler["type"]); a refit on either series would have re-introduced exactly the '
        'scale information this test must not use.',
        '',
        '## 9. Files',
        '',
        f'- Tables: `{tab_dir}/` (per_checkpoint_metrics, checkpoint_ratios, '
        'summary_by_model, weekly_ensemble, predictions_full_sequence).',
        f'- `tables/predictions_full_sequence.csv` stores EVERY point of EVERY '
        'checkpoint\'s walk-forward sequence (all 140 predicted weeks per checkpoint, '
        'with `in_eval_window` marking the scored 43 weeks).',
        f'- Figures: `{os.path.join(args.outdir, "figures")}/`.',
        f'- This report: `{os.path.join(args.outdir, "report_generalizability.md")}`.',
    ]
    report_path = os.path.join(args.outdir, 'report_generalizability.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    # ---- 7. Run manifest ----
    run_manifest = {
        'script': '14_c_generalizability_test.py',
        'args': {k: (str(v) if isinstance(v, pd.Timestamp) else v)
                 for k, v in vars(args).items()},
        'v01': {'path': args.v01, 'district': dist_a, 'n_weeks': len(weeks)},
        'v02': {'path': args.v02, 'district': dist_a, 'n_weeks': len(weeks_b)},
        'pre_window_identical': pre_identical,
        'eval_window': {'start': str(start.date()), 'end': str(end.date()),
                        'weeks': [str(w.date()) for w in eval_weeks]},
        'local_naive_denoms_v01': denoms_v01,
        'local_naive_denoms_v02': denoms_v02,
        'nb_nll': {'alpha_mode': args.alpha_mode,
                   'alpha_local_mom': alpha_local,
                   'alpha_mc_clean_train': alpha_mc,
                   'note': 'in sync with 15_error_metric.py; mom = shared '
                           'leakage-free alpha (apples-to-apples), profile = '
                           'per-scoring MLE (alpha_profile column)'},
        'clip_negative_forecasts': clip,
        'n_checkpoints': len(ckpts),
        'checkpoints': [r['checkpoint'] for r in ckpts],
    }
    with open(os.path.join(args.outdir, 'run_manifest.json'), 'w') as f:
        json.dump(run_manifest, f, indent=2)

    # ---- 8. Console summary ----
    print('\n=== Generalizability summary (eval window, both scored vs corrected truth v02) ===')
    show = summary[['model', 'scoring', 'rmse_mean', 'rel_rmse_mean',
                    'mase_m1_mean', 'rmsse_m1_mean', 'nb_nll_mean']]
    if len(show):
        print(show.round(3).to_string(index=False))

    print('\nMASE pred(v01)/pred(v02) median per model (>1 = under-reported inputs hurt):')
    for m, h in headline.items():
        print(f'  {m}: MASE {h["ratio_mase_med"]:.2f} | NLL improvement (corrected vs '
              f'under-reported inputs) median {h["nll_improvement_med"]:+.2f}% | '
              f'under-reported-input forecast worse in '
              f'{h["pct_v01_worse"]:.0f}% (MASE) / {h["pct_v01_worse_nll"]:.0f}% (NLL) '
              'of checkpoints')
    print(f'\nSaved outputs to: {args.outdir}')
    print('✓ Generalizability test complete.')


if __name__ == '__main__':
    main()
