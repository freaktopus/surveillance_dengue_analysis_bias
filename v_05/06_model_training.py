"""
06_model_training.py
--------------------
RNN/LSTM model training with bias propagation comparison.
Trains on 2022-2024, evaluates on 2025 holdout, scoring biased predictions
against the clean 2025 truth (apples-to-apples bias comparison).

Output (v_05/outputs/nbnll_model_training):
    model_results.csv             -- per-run metrics (incl. NB-NLL)
    model_predictions.csv         -- per-week obs vs pred
    bias_propagation_summary.csv  -- aggregated (incl. rel_rmse, cen_rmse, nb_nll)
"""

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os, warnings

warnings.filterwarnings('ignore')
torch.set_num_threads(4)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
OUT_DIR = os.path.join(BASE_DIR, 'nbnll_model_training')
os.makedirs(OUT_DIR, exist_ok=True)

N_DRAWS_SAMPLE = 100
N_SEEDS_BIASED = 3
N_SEEDS = 10

T_IN     = 8      # lookback window (weeks)
HIDDEN   = 16
EPOCHS   = 150
LR       = 0.001
BATCH    = 16

# Training window: 2022-2024; holdout: 2025
TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR   = 2025

np.random.seed(42)
print(f"06_model_training: draws={N_DRAWS_SAMPLE}, biased seeds={N_SEEDS_BIASED}, clean seeds={N_SEEDS}")

# Districts excluded from the study BY CHOICE (no training, no evaluation).
EXCLUDED_DISTRICTS = ['KASKI']

# ---- 1. LOAD DATA ------------------------------------------------------------------
clusters_df = pd.read_csv(f'{BASE_DIR}/district_clusters.csv')
weekly      = pd.read_csv(f'{BASE_DIR}/panel_weekly.csv')
clusters_df = clusters_df[~clusters_df['district'].isin(EXCLUDED_DISTRICTS)].copy()
weekly      = weekly[~weekly['district'].isin(EXCLUDED_DISTRICTS)].copy()
weekly['week_start'] = pd.to_datetime(weekly['week_start'])
weekly = weekly.merge(clusters_df[['district','cluster']], on='district')

# Cluster-aggregate clean panel
clean_agg = (weekly.groupby(['cluster','iso_year','iso_week','week_start'])
    ['weekly_cases'].sum().reset_index()
    .sort_values(['cluster','week_start']).reset_index(drop=True))

# Load biased panels
def load_biased(fname, n_draws=N_DRAWS_SAMPLE):
    df = pd.read_csv(f'{BASE_DIR}/{fname}')
    if 'district' in df.columns:
        df = df[~df['district'].isin(EXCLUDED_DISTRICTS)].copy()
    df['week_start'] = pd.to_datetime(df['week_start'])
    draws = sorted(df['draw'].unique())[:n_draws]
    df = df[df['draw'].isin(draws)]
    agg = (df.groupby(['cluster','iso_year','iso_week','week_start','draw'])
           ['biased_cases'].sum().reset_index()
           .sort_values(['cluster','draw','week_start']).reset_index(drop=True))
    return agg

print("Loading biased panels...")
b1 = load_biased('biased_B1.csv')
b2 = load_biased('biased_B2.csv')
b3 = load_biased('biased_B3.csv')
bc = load_biased('biased_combined.csv')
print("  Done.\n")

import torch.nn.functional as F
from scipy.special import gammaln


def nb_logpmf(y, mu, alpha):
    """Gamma-function extension of the NB pmf (size r = 1/alpha, mean mu),
    valid for real y >= 0 and mu > 0 (identical to common.py)."""
    r = 1.0 / max(alpha, 1e-10)
    p0 = r / (r + mu)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(p0) + y * np.log1p(-p0))


def nb_nll(true, pred, alpha):
    """Mean NB negative log-likelihood per observation, fixed alpha
    (identical to common.py nb_nll with leakage-free MoM alpha)."""
    t_ = np.maximum(np.asarray(true, dtype=float), 0.0)
    m_ = np.maximum(np.asarray(pred, dtype=float), 1e-8)
    return -float(np.mean(nb_logpmf(t_, m_, alpha)))


def nb_dispersion_mom(series, floor=1e-3):
    """Method-of-moments NB dispersion alpha from a count series
    (var = mu + alpha * mu^2); clipped to `floor`."""
    s = np.asarray(series, dtype=float)
    mu = s.mean()
    var = s.var(ddof=1) if len(s) > 1 else 0.0
    a = (var - mu) / max(mu ** 2, 1e-12)
    return float(max(a, floor))


# ---- 2. MODELS ---------------------------------------------------------------------
class VanillaRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ---- 3. DATA PREP ----------------------------------------------------------------
def make_sequences(series, t_in=T_IN):
    """Convert 1D series to (X, y) supervised sequences."""
    X, y = [], []
    for i in range(len(series) - t_in):
        X.append(series[i:i+t_in])
        y.append(series[i+t_in])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_eval(series_train, series_val, series_test, model_class, seed,
               true_test=None, ref_dates=None, alpha=None, test_dates=None):
    """Train model; return test predictions + metrics (scored vs CLEAN truth).

    Predictions are always generated from the input series (biased 2025 for biased
    conditions, clean 2025 for 'clean').  Metrics are scored against `true_test`
    when provided (clean 2025), else against `series_test` itself.
    NB-NLL uses the leakage-free per-cluster MoM alpha from the clean train panel.

    Scoring is DATE-ALIGNED on week_start (canonical evaluation window shared
    with the SEIR arm): prediction j forecasts input week T_IN+j, and is joined
    to the clean truth on that calendar week — never by array position. T_IN
    weeks are input context, not a redefinition of the scoring window.
    The 153-week 2022-2024 train split itself is calendar-consistent and kept.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Normalize by training mean+std
    mu  = series_train.mean() + 1e-8
    std = series_train.std()  + 1e-8
    def norm(x): return (x - mu) / std
    def denorm(x): return x * std + mu

    tr_n = norm(series_train)
    va_n = norm(series_val)
    te_n = norm(series_test)

    X_tr, y_tr = make_sequences(tr_n)
    X_va, y_va = make_sequences(va_n)
    X_te, _    = make_sequences(te_n)

    if len(X_tr) < 4:
        return None

    X_tr_t = torch.tensor(X_tr).unsqueeze(-1)
    y_tr_t = torch.tensor(y_tr)
    X_va_t = torch.tensor(X_va).unsqueeze(-1)
    y_va_t = torch.tensor(y_va)
    X_te_t = torch.tensor(X_te).unsqueeze(-1)

    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH, shuffle=True)

    model = model_class()
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None
    patience, patience_cnt = 20, 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_va_t)
            val_loss = loss_fn(val_pred, y_va_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        te_pred_n = model(X_te_t).numpy()
    te_pred = denorm(te_pred_n)

    # Reference truth for scoring — DATE-ALIGNED on week_start.
    # Prediction j forecasts the input week at position T_IN+j; join it to the
    # clean truth on that calendar week (inner one-to-one merge). A dropped or
    # duplicated week now fails loudly instead of silently shifting all pairs.
    ref = true_test if true_test is not None else series_test
    n   = min(len(te_pred), len(ref) - T_IN)
    if n < 1:
        return None
    te_pred = te_pred[:n]
    in_dates = (np.asarray(test_dates) if test_dates is not None
                else (np.asarray(ref_dates) if ref_dates is not None else None))
    if ref_dates is not None and in_dates is not None:
        pred_weeks = in_dates[T_IN:T_IN + n]
        truth_weeks = np.asarray(ref_dates)[T_IN:T_IN + n]
        ev = pd.DataFrame({'week_start': pd.to_datetime(truth_weeks),
                           'y_true': np.asarray(ref[T_IN:T_IN + n], dtype=float)})
        pr = pd.DataFrame({'week_start': pd.to_datetime(pred_weeks),
                           'y_pred': np.asarray(te_pred, dtype=float)})
        if ev['week_start'].duplicated().any() or pr['week_start'].duplicated().any():
            return None
        ev = ev.merge(pr, on='week_start', how='inner', validate='one_to_one')
        if len(ev) < 1:
            return None
        ev = ev.sort_values('week_start').reset_index(drop=True)
        te_true = ev['y_true'].values
        te_pred = ev['y_pred'].values
        weeks = ev['week_start'].values
    else:
        te_true = ref[T_IN:T_IN + n]
        weeks   = (ref_dates[T_IN:T_IN + n] if ref_dates is not None else np.arange(n))

    # ---- Level-dependent metrics ----
    rmse = np.sqrt(np.mean((te_true - te_pred)**2))
    mae  = np.mean(np.abs(te_true - te_pred))
    bias = np.mean(te_pred - te_true)              # signed mean error
    corr = np.corrcoef(te_true, te_pred)[0,1] if te_true.std() > 0 else 0.0

    # ---- Scale-free / level-robust metrics ----
    mse = rmse**2
    cen_rmse = float(np.sqrt(max(mse - bias**2, 0.0)))
    obs_mean = max(np.mean(te_true), 1e-8)
    rel_rmse = rmse / obs_mean                     # cross-cluster normaliser

    # ---- NB-NLL (uniform with 15_error_metric.py) ----
    nll = nb_nll(te_true, te_pred, alpha=alpha) if alpha is not None else np.nan

    return {'rmse': rmse, 'cen_rmse': cen_rmse, 'rel_rmse': rel_rmse,
            'mae': mae, 'forecast_bias': bias, 'corr': corr, 'nb_nll': nll,
            'predictions': te_pred, 'true': te_true, 'weeks': weeks}


# ---- 4. TRAIN-VAL-TEST SPLIT --------------------------------------------------------
TRAIN_FRAC = 0.70

# ---- 5. RUN ALL CONDITIONS ------------------------------------------------------------
CONDITIONS = {
    'clean':    (clean_agg,  None, 'weekly_cases'),
    'B1':       (b1,         None, 'biased_cases'),
    'B2':       (b2,         None, 'biased_cases'),
    'B3':       (b3,         None, 'biased_cases'),
    'combined': (bc,         None, 'biased_cases'),
}

MODEL_CLASSES = {'RNN': VanillaRNN, 'LSTM': LSTMModel}

results = []
pred_rows = []

for cluster in [1, 2, 3]:
    print(f"\n{'='*60}\nCluster {cluster}\n{'='*60}")

    clean_train_years = clean_agg[(clean_agg['cluster']==cluster) &
                            (clean_agg['iso_year'].isin(TRAIN_YEARS))].sort_values('week_start')
    clean_2025 = clean_agg[(clean_agg['cluster']==cluster) &
                            (clean_agg['iso_year']==TEST_YEAR)].sort_values('week_start')

    n_train_clean = int(len(clean_train_years) * TRAIN_FRAC)
    clean_train = clean_train_years['weekly_cases'].values[:n_train_clean]
    clean_val   = clean_train_years['weekly_cases'].values[n_train_clean:]
    clean_test  = clean_2025['weekly_cases'].values
    clean_dates = clean_2025['week_start'].values

    # Leakage-free NB dispersion: MoM alpha from the clean train window
    # (identical to 15_error_metric.py: full clean 2022-24 panel per cluster)
    alpha_cluster = nb_dispersion_mom(clean_train_years['weekly_cases'].values)
    print(f"  MoM alpha = {alpha_cluster:.4f}")

    for cond_name, (panel, _, case_col) in CONDITIONS.items():
        for model_name, model_class in MODEL_CLASSES.items():
            seed_metrics = []

            if cond_name == 'clean':
                for seed in range(N_SEEDS):
                    res = train_eval(clean_train, clean_val, clean_test, model_class, seed,
                                     true_test=clean_test, ref_dates=clean_dates,
                                     test_dates=clean_dates, alpha=alpha_cluster)
                    if res:
                        res['cluster'] = cluster
                        res['condition'] = cond_name
                        res['model'] = model_name
                        res['seed'] = seed
                        res['draw'] = -1
                        seed_metrics.append(res)
                        results.append({k: v for k, v in res.items()
                                       if k not in ['predictions','true','weeks']})
                        for w, o, p in zip(res['weeks'], res['true'], res['predictions']):
                            pred_rows.append({'cluster': cluster, 'model': model_name,
                                              'condition': cond_name, 'draw': -1, 'seed': seed,
                                              'week_start': str(w), 'observed': o, 'predicted': p})
            else:
                draws = sorted(panel['draw'].unique())
                for draw in draws[:N_DRAWS_SAMPLE]:
                    draw_train = panel[(panel['cluster']==cluster) &
                                       (panel['iso_year'].isin(TRAIN_YEARS)) &
                                       (panel['draw']==draw)].sort_values('week_start')
                    draw_2025 = panel[(panel['cluster']==cluster) &
                                      (panel['iso_year']==TEST_YEAR) &
                                      (panel['draw']==draw)].sort_values('week_start')

                    if len(draw_train) < 10:
                        continue
                    if len(clean_test) < T_IN + 2:
                        continue

                    biased_train = draw_train[case_col].values[:n_train_clean]
                    biased_val   = draw_train[case_col].values[n_train_clean:]
                    biased_test  = draw_2025[case_col].values

                    if len(biased_test) < T_IN + 2:
                        continue

                    for seed in range(N_SEEDS_BIASED):
                        # Score biased predictions against the CLEAN 2025 truth
                        res = train_eval(biased_train, biased_val, biased_test,
                                         model_class, seed, true_test=clean_test,
                                         ref_dates=clean_dates,
                                         test_dates=draw_2025['week_start'].values,
                                         alpha=alpha_cluster)
                        if res:
                            res['cluster'] = cluster
                            res['condition'] = cond_name
                            res['model'] = model_name
                            res['seed'] = seed
                            res['draw'] = draw
                            seed_metrics.append(res)
                            results.append({k: v for k, v in res.items()
                                           if k not in ['predictions','true','weeks']})
                            for w, o, p in zip(res['weeks'], res['true'], res['predictions']):
                                pred_rows.append({'cluster': cluster, 'model': model_name,
                                                  'condition': cond_name, 'draw': draw, 'seed': seed,
                                                  'week_start': str(w), 'observed': o, 'predicted': p})

            if seed_metrics:
                rmses   = [r['rmse'] for r in seed_metrics]
                rels    = [r['rel_rmse'] for r in seed_metrics]
                biases  = [r['forecast_bias'] for r in seed_metrics]
                corrs   = [r['corr'] for r in seed_metrics]
                nlls    = [r['nb_nll'] for r in seed_metrics if not np.isnan(r['nb_nll'])]
                nll_str = f" | NBNLL={np.mean(nlls):.3f}" if nlls else ""
                print(f"  {model_name} [{cond_name}]: RMSE={np.mean(rmses):.2f}±{np.std(rmses):.2f} | "
                      f"relRMSE={np.mean(rels):.3f} | Bias={np.mean(biases):+.2f} | "
                      f"r={np.mean(corrs):.3f}{nll_str}")

# ---- 6. SAVE RESULTS ----------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f'{OUT_DIR}/model_results.csv', index=False)
pd.DataFrame(pred_rows).to_csv(f'{OUT_DIR}/model_predictions.csv', index=False)
print(f"\nSaved: {OUT_DIR}/model_results.csv ({len(results_df)} rows)")
print(f"Saved: {OUT_DIR}/model_predictions.csv ({len(pred_rows)} rows)")

# ---- 7. BIAS PROPAGATION SUMMARY --------------------------------------------------------
summary_rows = []

for cluster in [1, 2, 3]:
    for model in ['RNN', 'LSTM']:
        clean_df = results_df[(results_df['cluster']==cluster) &
                               (results_df['model']==model) &
                               (results_df['condition']=='clean')]
        if len(clean_df) == 0:
            continue
        baseline_rmse = clean_df['rmse'].mean()
        baseline_rel  = clean_df['rel_rmse'].mean()
        baseline_mae  = clean_df['mae'].mean()
        baseline_nll  = clean_df['nb_nll'].mean()

        for cond in ['B1','B2','B3','combined']:
            cdf = results_df[(results_df['cluster']==cluster) &
                               (results_df['model']==model) &
                               (results_df['condition']==cond)]
            if len(cdf) == 0:
                continue
            mean_rmse = cdf['rmse'].mean()
            ci95_rmse = 1.96 * cdf['rmse'].std() / np.sqrt(len(cdf))
            mean_bias = cdf['forecast_bias'].mean()
            mean_corr = cdf['corr'].mean()
            mean_rel  = cdf['rel_rmse'].mean()
            mean_cen  = cdf['cen_rmse'].mean()
            mean_nll  = cdf['nb_nll'].mean()

            summary_rows.append({
                'cluster': cluster, 'model': model, 'condition': cond,
                'baseline_rmse': baseline_rmse,
                'biased_rmse': mean_rmse,
                'rmse_ci95': ci95_rmse,
                'rmse_degradation_abs': mean_rmse - baseline_rmse,
                'rmse_degradation_pct': (mean_rmse - baseline_rmse) / baseline_rmse * 100,
                'baseline_rel_rmse': baseline_rel,
                'biased_rel_rmse': mean_rel,
                'rel_rmse_degradation_pct': (mean_rel - baseline_rel) / baseline_rel * 100,
                'cen_rmse_mean': mean_cen,
                'baseline_nb_nll': baseline_nll,
                'nb_nll_mean': mean_nll,
                'nll_degradation_abs': mean_nll - baseline_nll,
                'nll_degradation_pct': (mean_nll - baseline_nll) / abs(baseline_nll) * 100,
                'mean_forecast_bias': mean_bias,
                'mean_corr': mean_corr,
                'n_observations': len(cdf)
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{OUT_DIR}/bias_propagation_summary.csv', index=False)
print(f"Saved: {OUT_DIR}/bias_propagation_summary.csv")

print("\n=== Bias Propagation Summary (level RMSE vs scale-free relRMSE vs NB-NLL) ===")
for model in ['RNN', 'LSTM']:
    print(f"\n  {model}:")
    sub = summary_df[summary_df['model']==model]
    pivot = sub.pivot_table(
        index=['cluster','condition'],
        values=['rmse_degradation_pct','rel_rmse_degradation_pct',
                'nb_nll_mean','nll_degradation_pct','mean_forecast_bias','mean_corr'],
        aggfunc='mean'
    )
    print(pivot.round(2).to_string())

print("\n✓ Model training complete.")
