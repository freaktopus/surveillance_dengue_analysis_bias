import argparse
import json
import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import gammaln

warnings.filterwarnings('ignore')
torch.set_num_threads(4)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
OUT_DIR = os.path.join(BASE_DIR, 'nbnll_13_retrain_cluster2')
WEIGHT_DIR = os.path.join(OUT_DIR, 'weights_cluster2_retrain')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(WEIGHT_DIR, exist_ok=True)

# ---- Config (architecture/hyperparameters identical to 06_model_training.py) ----
CLUSTER = 2
CLUSTER_LABEL = 'Peri-urban / Mid Surveillance'
T_IN = 8            # lookback window length (weeks) -- saved with every model
HIDDEN = 16
EPOCHS = 150
LR = 0.001
BATCH = 16
TRAIN_FRAC = 0.70   # first 70% of the TRAIN window -> train, rest -> val

N_DRAWS_SAMPLE = 100
N_SEEDS_BIASED = 3
N_SEEDS = 10

# Hard temporal cutoff: train ONLY on weeks <= 2024-10-21 (ISO 2024-W43, Monday).
# Everything after is prediction-only (forward pass).
TRAIN_END = pd.Timestamp('2024-10-21')
TRAIN_END_ISO = f"{TRAIN_END.isocalendar().year}-W{TRAIN_END.isocalendar().week:02d}"

# Districts excluded from the study BY CHOICE (no training, no evaluation).
EXCLUDED_DISTRICTS = ['KASKI']

# Univariate input feature (biased series use the biased case column, same role).
FEATURE_ORDER = ['weekly_cases']
TARGET = 'weekly_cases'   # one-step-ahead target

CONDITION_FILES = {
    'clean': None,
    'B1': 'biased_B1.csv',
    'B2': 'biased_B2.csv',
    'B3': 'biased_B3.csv',
    'combined': 'biased_combined.csv',
}

# ---- Normalization transforms -------------------------------------------------
# The ACTIVE transform is recorded in every checkpoint as scaler['type'] (and
# mirrored in manifest.json).  Consumers MUST dispatch on that field:
#   'zscore' -> use scaler['mu']/scaler['std']:  (x - mu)/std  ...  y*std + mu
#   'log1p'  -> use log1p(x) ... expm1(y);  scaler['mu']/['std'] are None
#               (descriptive raw-split stats kept under raw_split_mu/raw_split_std)
# Never assume 'zscore' -- older/mismatched labels silently explode forward passes.
DEFAULT_TRANSFORM = 'log1p'
TRANSFORMS = {
    'zscore': {'norm': lambda x, mu, std: (x - mu) / std,
               'denorm': lambda y, mu, std: y * std + mu,
               'norm_name': '(x - mu) / std', 'denorm_name': 'y * std + mu'},
    'log1p': {'norm': lambda x, mu, std: np.log1p(x),
              'denorm': lambda y, mu, std: np.expm1(y),
              'norm_name': 'log1p(x)', 'denorm_name': 'expm1(y)'},
}


# ---- Models (mirror 06_model_training.py) ----
class VanillaRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


MODEL_CLASSES = {'RNN': VanillaRNN, 'LSTM': LSTMModel}


# ---- NB-NLL scoring (mirrors 15_error_metric.py / common.py) ----------------
def nb_logpmf(y, mu, alpha):
    r = 1.0 / max(alpha, 1e-10)
    p0 = r / (r + mu)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(p0) + y * np.log1p(-p0))


def nb_nll(true, pred, alpha):
    t_ = np.maximum(np.asarray(true, dtype=float), 0.0)
    m_ = np.maximum(np.asarray(pred, dtype=float), 1e-8)
    return -float(np.mean(nb_logpmf(t_, m_, alpha)))


def nb_dispersion_mom(series, floor=1e-3):
    s = np.asarray(series, dtype=float)
    mu = s.mean()
    var = s.var(ddof=1) if len(s) > 1 else 0.0
    a = (var - mu) / max(mu ** 2, 1e-12)
    return float(max(a, floor))


def make_sequences(series, t_in=T_IN):
    """Convert 1D series to (X, y) supervised sequences (same as 06)."""
    X, y = [], []
    for i in range(len(series) - t_in):
        X.append(series[i:i + t_in])
        y.append(series[i + t_in])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_eval(train_window, pred_series, model_class, seed,
               true_pred=None, ref_dates=None, epochs=EPOCHS,
               transform=DEFAULT_TRANSFORM, alpha=None):
    """Train on `train_window` (<= cutoff), forward-pass over `pred_series`.

    train_window : full training-window values (weeks <= TRAIN_END). Internally
                   split 70/30 into train/val; the normalizer selected by
                   `transform` (see TRANSFORMS) is fixed for train/val/forward
                   pass and its NAME is returned for saving.
    pred_series  : post-cutoff values of the SAME condition (model input).
    true_pred    : clean post-cutoff truth for scoring (apples-to-apples).
    ref_dates    : week_start dates of the post-cutoff window.

    Returns metrics + per-week predictions + state_dict + scaler spec, or None.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_tr = int(len(train_window) * TRAIN_FRAC)
    series_train = train_window[:n_tr]
    series_val = train_window[n_tr:]

    # Normalizer fixed at training time; its NAME is saved with the checkpoint.
    # mu/std describe the raw training split (required by 'zscore'; kept as
    # descriptive raw_split_* metadata for 'log1p').
    mu = float(series_train.mean() + 1e-8)
    std = float(series_train.std() + 1e-8)
    tf = TRANSFORMS[transform]
    norm = lambda x: tf['norm'](x, mu, std)
    denorm = lambda y: tf['denorm'](y, mu, std)

    X_tr, y_tr = make_sequences(norm(series_train))
    X_va, y_va = make_sequences(norm(series_val))
    if len(X_tr) < 4 or len(X_va) < 1:
        return None

    X_tr_t = torch.tensor(X_tr).unsqueeze(-1)
    y_tr_t = torch.tensor(y_tr)
    X_va_t = torch.tensor(X_va).unsqueeze(-1)
    y_va_t = torch.tensor(y_va)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH, shuffle=True)

    model = model_class()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None
    patience, patience_cnt = 20, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_state is None:
        return None
    model.load_state_dict(best_state)
    model.eval()

    # ---- Pure one-step-ahead FORWARD PASS over the post-cutoff window ----
    # Context: last T_IN training weeks seed the first post-cutoff prediction,
    # so EVERY week after the cutoff is predicted (nothing is dropped).
    full = np.concatenate([train_window, np.asarray(pred_series, dtype=float)])
    full_n = norm(full)
    X_all, _ = make_sequences(full_n)                 # seq i -> target full[i+T_IN]
    start = len(train_window) - T_IN                  # first target = first post-cutoff week
    X_fwd = torch.tensor(X_all[start:]).unsqueeze(-1) # targets = all post-cutoff weeks

    with torch.no_grad():
        fwd_pred = denorm(model(X_fwd).numpy())

    ref = true_pred if true_pred is not None else np.asarray(pred_series, dtype=float)
    n = min(len(fwd_pred), len(ref))
    if n < 1:
        return None
    fwd_pred = fwd_pred[:n]
    fwd_true = ref[:n]
    weeks = (ref_dates[:n] if ref_dates is not None else np.arange(n))

    # ---- Level-dependent metrics (scored vs clean truth) ----
    rmse = float(np.sqrt(np.mean((fwd_true - fwd_pred) ** 2)))
    mae = float(np.mean(np.abs(fwd_true - fwd_pred)))
    bias = float(np.mean(fwd_pred - fwd_true))
    corr = float(np.corrcoef(fwd_true, fwd_pred)[0, 1]) if fwd_true.std() > 0 else 0.0

    # ---- Scale-free / level-robust metrics ----
    mse = rmse ** 2
    cen_rmse = float(np.sqrt(max(mse - bias ** 2, 0.0)))
    obs_mean = max(float(np.mean(fwd_true)), 1e-8)
    rel_rmse = float(rmse / obs_mean)

    # ---- NB-NLL (uniform with 15_error_metric.py) ----
    nll = nb_nll(fwd_true, fwd_pred, alpha=alpha) if alpha is not None else np.nan

    return {'rmse': rmse, 'cen_rmse': cen_rmse, 'rel_rmse': rel_rmse,
            'mae': mae, 'forecast_bias': bias, 'corr': corr, 'nb_nll': nll,
            'best_val_loss': float(best_val_loss),
            'predictions': fwd_pred, 'true': fwd_true, 'weeks': weeks,
            'state_dict': best_state, 'mu': mu, 'std': std,
            'transform': transform,
            'n_train_split': int(n_tr), 'n_val_split': int(len(series_val))}


def load_biased(fname, n_draws):
    df = pd.read_csv(os.path.join(BASE_DIR, fname))
    if 'district' in df.columns:
        df = df[~df['district'].isin(EXCLUDED_DISTRICTS)].copy()
    df['week_start'] = pd.to_datetime(df['week_start'])
    draws = sorted(df['draw'].unique())[:n_draws]
    df = df[df['draw'].isin(draws)]
    return (df.groupby(['cluster', 'iso_year', 'iso_week', 'week_start', 'draw'])
              ['biased_cases'].sum().reset_index()
              .sort_values(['cluster', 'draw', 'week_start']).reset_index(drop=True))


def save_checkpoint(model_name, cond_name, draw, seed, res, epochs):
    """Save weights + architecture + the EXACT normalization spec (scaler['type']
    records the transform consumers must dispatch on; mu/std are only meaningful
    for 'zscore' and are None otherwise)."""
    transform = res['transform']
    tf = TRANSFORMS[transform]
    is_z = transform == 'zscore'
    fname = f"{model_name}_{cond_name}_draw{draw}_seed{seed}.pt"
    path = os.path.join(WEIGHT_DIR, fname)
    torch.save({
        # -- architecture (rebuild the exact model) --
        'arch': {'name': model_name, 'input_size': 1,
                 'hidden': HIDDEN, 't_in': T_IN},
        'state_dict': res['state_dict'],
        # -- exact normalization spec (DISPATCH on 'type') --
        'scaler': {'type': transform,
                   'norm': tf['norm_name'], 'denorm': tf['denorm_name'],
                   'mu': res['mu'] if is_z else None,
                   'std': res['std'] if is_z else None,
                   'raw_split_mu': res['mu'], 'raw_split_std': res['std'],
                   'fitted_on': f'training split (first {TRAIN_FRAC:.0%} of weeks '
                                f'<= {TRAIN_END.date()}); '
                                + ('mu/std fitted on this split' if is_z else
                                   'mu/std above are descriptive raw-split stats only')},
        # -- feature / window spec needed for a forward pass --
        'feature_order': FEATURE_ORDER,
        'target': TARGET,
        # -- bookkeeping --
        'cluster': CLUSTER,
        'cluster_label': CLUSTER_LABEL,
        'condition': cond_name,
        'draw': int(draw),
        'seed': int(seed),
        'lr': LR,
        'batch': BATCH,
        'epochs_run': epochs,
        'train_end': str(TRAIN_END.date()),
        'train_end_iso_week': TRAIN_END_ISO,
        'excluded_districts': EXCLUDED_DISTRICTS,
        'metrics': {k: res[k] for k in
                    ('rmse', 'cen_rmse', 'rel_rmse', 'mae', 'forecast_bias',
                     'corr', 'nb_nll')},
    }, path)
    return fname


def main():
    ap = argparse.ArgumentParser(
        description='Cluster-2 retraining with hard cutoff 2024-10-21 (ISO 2024-W43); '
                    'post-cutoff = forward pass only. Saves weights+arch+scaler+spec.')
    ap.add_argument('--conditions', nargs='+',
                    default=['clean', 'B1', 'B2', 'B3', 'combined'])
    ap.add_argument('--models', nargs='+', default=['RNN', 'LSTM'])
    ap.add_argument('--n-draws', type=int, default=N_DRAWS_SAMPLE)
    ap.add_argument('--seeds-clean', type=int, default=N_SEEDS)
    ap.add_argument('--seeds-biased', type=int, default=N_SEEDS_BIASED)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--transform', choices=sorted(TRANSFORMS), default=DEFAULT_TRANSFORM,
                    help='input/target normalization; saved as scaler["type"] in every '
                         'checkpoint and dispatch key for consumers (default: log1p)')
    args = ap.parse_args()

    conditions = [c for c in args.conditions if c in CONDITION_FILES]
    models = [m for m in args.models if m in MODEL_CLASSES]
    if not conditions:
        raise SystemExit(f"No valid conditions: {args.conditions}")
    if not models:
        raise SystemExit(f"No valid models: {args.models}")
    missing = [c for c in conditions if c != 'clean'
               and not os.path.exists(os.path.join(BASE_DIR, CONDITION_FILES[c]))]
    if missing:
        raise SystemExit(f"Missing biased panels for: {missing}. Run 01-04 first.")

    np.random.seed(42)
    print(f"13_retrain: cluster={CLUSTER} ({CLUSTER_LABEL}), models={models}, "
          f"conditions={conditions}, draws={args.n_draws}, "
          f"seeds clean/biased={args.seeds_clean}/{args.seeds_biased}, epochs={args.epochs}, "
          f"transform={args.transform}")
    print(f"Train window: panel start -> {TRAIN_END.date()} (ISO {TRAIN_END_ISO}); "
          f"prediction: all later weeks. Excluded: {EXCLUDED_DISTRICTS}")

    # ---- Load clean cluster aggregate (KASKI excluded) ----
    clusters_df = pd.read_csv(os.path.join(BASE_DIR, 'district_clusters.csv'))
    weekly = pd.read_csv(os.path.join(BASE_DIR, 'panel_weekly.csv'))
    clusters_df = clusters_df[~clusters_df['district'].isin(EXCLUDED_DISTRICTS)].copy()
    weekly = weekly[~weekly['district'].isin(EXCLUDED_DISTRICTS)].copy()
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    weekly = weekly.merge(clusters_df[['district', 'cluster']], on='district')
    clean_agg = (weekly.groupby(['cluster', 'iso_year', 'iso_week', 'week_start'])
                 ['weekly_cases'].sum().reset_index()
                 .sort_values(['cluster', 'week_start']).reset_index(drop=True))

    panels = {}
    for c in conditions:
        if c == 'clean':
            panels[c] = None
        else:
            panels[c] = load_biased(CONDITION_FILES[c], args.n_draws)
            print(f"  loaded {c}: {len(panels[c])} rows")

    # ---- Cluster-2 train/predict split at the hard cutoff ----
    c2 = clean_agg[clean_agg['cluster'] == CLUSTER].sort_values('week_start')
    clean_train_df = c2[c2['week_start'] <= TRAIN_END]
    clean_pred_df = c2[c2['week_start'] > TRAIN_END]
    if len(clean_train_df) < T_IN + 10 or len(clean_pred_df) < 2:
        raise SystemExit('Insufficient cluster-2 weeks around the cutoff.')

    clean_train = clean_train_df['weekly_cases'].values
    clean_pred = clean_pred_df['weekly_cases'].values
    pred_dates = clean_pred_df['week_start'].values

    iso0 = clean_train_df.iloc[0]
    print(f"Cluster {CLUSTER}: train weeks={len(clean_train)} "
          f"(ISO {int(iso0['iso_year'])}-W{int(iso0['iso_week']):02d} .. {TRAIN_END_ISO}), "
          f"predict weeks={len(clean_pred)} "
          f"({pd.Timestamp(pred_dates[0]).date()} .. {pd.Timestamp(pred_dates[-1]).date()})")

    # Leakage-free NB dispersion: MoM alpha from the clean cluster-2 train window
    # (identical to 15_error_metric.py; no post-cutoff data used)
    alpha_cluster = nb_dispersion_mom(clean_train)
    print(f"  MoM alpha = {alpha_cluster:.4f}")

    results, pred_rows, manifest_runs = [], [], []

    for cond_name in conditions:
        panel = panels[cond_name]
        for model_name in models:
            model_class = MODEL_CLASSES[model_name]
            seed_metrics = []

            def record(res, draw, seed):
                fname = save_checkpoint(model_name, cond_name, draw, seed, res, args.epochs)
                seed_metrics.append(res)
                results.append({'cluster': CLUSTER, 'cluster_label': CLUSTER_LABEL,
                                'condition': cond_name, 'model': model_name,
                                'seed': seed, 'draw': int(draw),
                                'rmse': res['rmse'], 'cen_rmse': res['cen_rmse'],
                                'rel_rmse': res['rel_rmse'], 'mae': res['mae'],
                                'forecast_bias': res['forecast_bias'], 'corr': res['corr'],
                                'nb_nll': res['nb_nll'],
                                'best_val_loss': res['best_val_loss'],
                                'n_train_split': res['n_train_split'],
                                'n_val_split': res['n_val_split'],
                                'n_pred_weeks': len(res['predictions'])})
                manifest_runs.append({'checkpoint': fname, 'model': model_name,
                                      'condition': cond_name, 'draw': int(draw),
                                      'seed': seed, 'mu': res['mu'], 'std': res['std'],
                                      'rmse': res['rmse'], 'mae': res['mae'],
                                      'forecast_bias': res['forecast_bias'],
                                      'corr': res['corr'],
                                      'nb_nll': res['nb_nll']})
                for w, o, p in zip(res['weeks'], res['true'], res['predictions']):
                    pred_rows.append({'cluster': CLUSTER, 'model': model_name,
                                      'condition': cond_name, 'draw': int(draw),
                                      'seed': seed, 'week_start': str(w),
                                      'observed': o, 'predicted': p})

            if cond_name == 'clean':
                for seed in range(args.seeds_clean):
                    res = train_eval(clean_train, clean_pred, model_class, seed,
                                     true_pred=clean_pred, ref_dates=pred_dates,
                                     epochs=args.epochs, transform=args.transform,
                                     alpha=alpha_cluster)
                    if res:
                        record(res, -1, seed)
            else:
                psub = panel[panel['cluster'] == CLUSTER]
                for draw in sorted(psub['draw'].unique())[:args.n_draws]:
                    dser = psub[psub['draw'] == draw].sort_values('week_start')
                    d_train = dser[dser['week_start'] <= TRAIN_END]['biased_cases'].values
                    d_pred = dser[dser['week_start'] > TRAIN_END]['biased_cases'].values
                    if len(d_train) < T_IN + 10 or len(d_pred) < 2:
                        continue
                    for seed in range(args.seeds_biased):
                        # Score biased forward-pass against the CLEAN truth
                        res = train_eval(d_train, d_pred, model_class, seed,
                                         true_pred=clean_pred, ref_dates=pred_dates,
                                         epochs=args.epochs, transform=args.transform,
                                         alpha=alpha_cluster)
                        if res:
                            record(res, draw, seed)

            if seed_metrics:
                rmses = [r['rmse'] for r in seed_metrics]
                maes = [r['mae'] for r in seed_metrics]
                biases = [r['forecast_bias'] for r in seed_metrics]
                corrs = [r['corr'] for r in seed_metrics]
                nlls = [r['nb_nll'] for r in seed_metrics if not np.isnan(r['nb_nll'])]
                nll_str = f" | NBNLL={np.mean(nlls):.3f}" if nlls else ""
                print(f"  {model_name} [{cond_name}]: n={len(seed_metrics)} "
                      f"RMSE={np.mean(rmses):.2f}±{np.std(rmses):.2f} | "
                      f"MAE={np.mean(maes):.2f} | Bias={np.mean(biases):+.2f} | "
                      f"r={np.mean(corrs):.3f}{nll_str}")

    # ---- Save results / predictions ----
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUT_DIR, 'retrain_results.csv'), index=False)
    pd.DataFrame(pred_rows).to_csv(os.path.join(OUT_DIR, 'retrain_predictions.csv'), index=False)
    print(f"\nSaved: {OUT_DIR}/retrain_results.csv ({len(results_df)} rows)")
    print(f"Saved: {OUT_DIR}/retrain_predictions.csv ({len(pred_rows)} rows)")

    # ---- Aggregated error summary ----
    summary_rows = []
    for model_name in models:
        for cond_name in conditions:
            sub = results_df[(results_df['model'] == model_name) &
                             (results_df['condition'] == cond_name)]
            if len(sub) == 0:
                continue
            summary_rows.append({
                'cluster': CLUSTER, 'model': model_name, 'condition': cond_name,
                'n_runs': len(sub),
                'rmse_mean': sub['rmse'].mean(), 'rmse_std': sub['rmse'].std(),
                'rmse_ci95': 1.96 * sub['rmse'].std() / np.sqrt(len(sub)),
                'mae_mean': sub['mae'].mean(),
                'rel_rmse_mean': sub['rel_rmse'].mean(),
                'cen_rmse_mean': sub['cen_rmse'].mean(),
                'nb_nll_mean': sub['nb_nll'].mean(),
                'forecast_bias_mean': sub['forecast_bias'].mean(),
                'corr_mean': sub['corr'].mean(),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, 'retrain_summary.csv'), index=False)
    print(f"Saved: {OUT_DIR}/retrain_summary.csv")

    # ---- Manifest: global spec + index of every checkpoint ----
    manifest = {
        'script': '13_retrain.py',
        'purpose': 'cluster-2 retraining, hard cutoff; post-cutoff = forward pass',
        'cluster': CLUSTER,
        'cluster_label': CLUSTER_LABEL,
        'excluded_districts': EXCLUDED_DISTRICTS,
        'train_window': {'start': str(clean_train_df['week_start'].iloc[0].date()),
                         'end': str(TRAIN_END.date()),
                         'end_iso_week': TRAIN_END_ISO,
                         'n_weeks': int(len(clean_train)),
                         'train_frac': TRAIN_FRAC},
        'predict_window': {'start': str(pd.Timestamp(pred_dates[0]).date()),
                           'end': str(pd.Timestamp(pred_dates[-1]).date()),
                           'n_weeks': int(len(clean_pred)),
                           'mode': 'one-step-ahead forward pass '
                                   '(last T_IN train weeks seed the context)'},
        'arch': {'models': models, 'input_size': 1, 'hidden': HIDDEN,
                 't_in': T_IN, 'epochs': args.epochs, 'lr': LR, 'batch': BATCH,
                 'patience': 20, 'grad_clip': 1.0, 'loss': 'MSELoss',
                 'optimizer': 'Adam'},
        'scaler': {'type': args.transform,
                   'norm': TRANSFORMS[args.transform]['norm_name'],
                   'denorm': TRANSFORMS[args.transform]['denorm_name'],
                   'note': 'per-checkpoint scaler in each .pt; DISPATCH on '
                           'scaler["type"]: zscore -> use mu/std; log1p -> use '
                           'log1p/expm1 (mu/std are None; raw-split stats under '
                           'raw_split_mu/raw_split_std)'},
        'feature_order': FEATURE_ORDER,
        'target': TARGET,
        'scoring': 'biased conditions scored against CLEAN post-cutoff truth; '
                   'all runs also scored with NB-NLL (leakage-free MoM alpha '
                   'from the clean train window, uniform with 15_error_metric.py)',
        'weight_dir': WEIGHT_DIR,
        'n_checkpoints': len(manifest_runs),
        'runs': manifest_runs,
    }
    with open(os.path.join(OUT_DIR, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved: {OUT_DIR}/manifest.json")
    print(f"Saved: {WEIGHT_DIR}/ ({len(manifest_runs)} checkpoints)")

    print("\n=== Retrain Summary (post-cutoff forward pass, cluster 2) ===")
    if len(summary_df):
        print(summary_df.pivot_table(index='condition', columns='model',
                                     values=['rmse_mean', 'nb_nll_mean']).round(2).to_string())
    print("\n✓ Retraining complete.")


if __name__ == '__main__':
    main()
