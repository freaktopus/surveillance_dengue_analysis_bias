"""
v_05/common.py
----------------
Ported from v_02/common: centralised data loaders, models, metrics, and
stage-1 corrections adapted for the v_05 pipeline (outputs → v_05/outputs).

Leakage policy (operational arm only, no leakage):
  - May read: cluster prior params from bias_parameters.csv,
            biased training series from biased_B1/B2/B3/combined.csv,
            district clusters from district_clusters.csv.
  - MUST NOT read: clean test series (2025), per-draw TRUE rates from
    bias_draw_params.csv, clean reference for cross-correlation lag.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

_V05_DIR = os.path.dirname(os.path.abspath(__file__))     # v_05/
_OUT_DIR = os.path.join(_V05_DIR, 'outputs')             # v_05/outputs
os.makedirs(_OUT_DIR, exist_ok=True)

# ── Paths ──────────────────────────────────────────────────────────────
REPO_DIR = os.path.dirname(_V05_DIR)                      # repo root
OUT_DIR  = os.path.join(REPO_DIR, 'outputs')              # repo-root outputs (read-only for v_01 artifacts)
V05_OUT  = _OUT_DIR                                        # v_05/outputs (primary pipeline dir)

TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR   = 2025
CONDITIONS  = ['B1', 'B2', 'B3', 'combined']

# Districts excluded from the study BY CHOICE: they take no part in clustering,
# panel aggregation, bias injection, model training, or filter evaluation.
EXCLUDED_DISTRICTS = ['KASKI']


def drop_excluded_districts(df, col='district'):
    """Remove excluded districts (e.g. KASKI) from any district-level frame."""
    if df is None or len(df) == 0 or col not in df.columns:
        return df
    return df[~df[col].isin(EXCLUDED_DISTRICTS)].copy()

CLUSTER_LABELS = {
    1: 'Urban / High Surveillance',
    2: 'Peri-urban / Mid Surveillance',
    3: 'Rural / Low Surveillance',
}

# ── Data loaders ──────────────────────────────────────────────────────
def load_bias_params():
    """Cluster-level prior parameters (Beta/Gamma). Operational-safe: returns means only."""
    return pd.read_csv(os.path.join(V05_OUT, 'bias_parameters.csv')).set_index('cluster').to_dict('index')


def load_draw_params():
    """Per-district per-draw TRUE sampled bias rates/lags. ORACLE-ONLY — operational arm must NOT use."""
    path = os.path.join(V05_OUT, 'bias_draw_params.csv')
    if os.path.exists(path):
        return drop_excluded_districts(pd.read_csv(path))
    return pd.DataFrame()


def load_biased(btype, clusters=None, draws=None, n_draws=None):
    """Return the cluster-aggregated biased panel for a bias type.

    btype   : 'B1' | 'B2' | 'B3' | 'combined'
    n_draws : if set, take only the first n_draws draws (for cheap smoke runs)
    draws   : explicit draw list (overrides n_draws)
    """
    fname = {'B1': 'biased_B1.csv', 'B2': 'biased_B2.csv',
             'B3': 'biased_B3.csv', 'combined': 'biased_combined.csv'}[btype]
    fpath = os.path.join(V05_OUT, fname)
    if not os.path.exists(fpath):
        return pd.DataFrame()
    df = pd.read_csv(fpath)
    df['week_start'] = pd.to_datetime(df['week_start'])
    df = drop_excluded_districts(df)
    if clusters is not None:
        df = df[df['cluster'].isin(clusters)]
    if draws is not None:
        df = df[df['draw'].isin(draws)]
    elif n_draws is not None:
        keep = sorted(df['draw'].unique())[:n_draws]
        df = df[df['draw'].isin(keep)]
    agg = (df.groupby(['cluster', 'iso_year', 'iso_week', 'week_start', 'draw'])
           ['biased_cases'].sum().reset_index()
           .sort_values(['cluster', 'draw', 'week_start']).reset_index(drop=True))
    return agg


def load_cluster_clean():
    """Cluster-aggregated CLEAN weekly cases, split-friendly (for diagnostics only)."""
    clusters_df = pd.read_csv(os.path.join(V05_OUT, 'district_clusters.csv'))
    weekly = pd.read_csv(os.path.join(V05_OUT, 'panel_weekly.csv'))
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    clusters_df = drop_excluded_districts(clusters_df)
    weekly = drop_excluded_districts(weekly)
    weekly = weekly.merge(clusters_df[['district', 'cluster']], on='district')
    agg = (weekly.groupby(['cluster', 'iso_year', 'iso_week', 'week_start'])
           ['weekly_cases'].sum().reset_index()
           .sort_values(['cluster', 'week_start']).reset_index(drop=True))
    return agg, clusters_df


def per_cluster_years(clean_agg, cluster, train_years=None, test_year=None):
    """Return clean per-year series for cluster. Used for train/val/test split design."""
    if train_years is None:
        train_years = TRAIN_YEARS
    if test_year is None:
        test_year = TEST_YEAR
    tr = clean_agg[(clean_agg['cluster'] == cluster) & (clean_agg['iso_year'].isin(train_years))]
    te = clean_agg[(clean_agg['cluster'] == cluster) & (clean_agg['iso_year'] == test_year)]
    return (tr.sort_values('week_start').reset_index(drop=True),
            te.sort_values('week_start').reset_index(drop=True))


# ── Models ─────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class ModelConfig:
    t_in: int = 8
    hidden: int = 16
    epochs: int = 80
    lr: float = 0.001
    batch: int = 16
    patience: int = 15


class VanillaRNN(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.rnn = nn.RNN(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class LSTMModel(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


MODELS = {'RNN': VanillaRNN, 'LSTM': LSTMModel}

def make_sequences(series, t_in=8):
    X, y = [], []
    for i in range(len(series) - t_in):
        X.append(series[i:i + t_in])
        y.append(series[i + t_in])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def fit_model(train, val, ModelCls, seed, cfg=ModelConfig()):
    """Train (normalised) RNN/LSTM with early stopping on val. Returns model, mu, sd."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    mu, sd = train.mean() + 1e-8, train.std() + 1e-8
    norm = lambda x: (x - mu) / sd
    Xtr, ytr = make_sequences(norm(train), cfg.t_in)
    Xva, yva = make_sequences(norm(val), cfg.t_in)
    if len(Xtr) < 3:
        return None, None, None
    loader = DataLoader(TensorDataset(torch.tensor(Xtr).unsqueeze(-1),
                                      torch.tensor(ytr)),
                        batch_size=cfg.batch, shuffle=True)
    model = ModelCls(cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()
    best_val, best_state, pat = np.inf, None, 0
    model.train()
    for ep in range(cfg.epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(torch.tensor(Xva).unsqueeze(-1)),
                         torch.tensor(yva)).item()
        if vl < best_val:
            best_val = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return model, mu, sd


def predict(model, series, mu, sd, t_in=8, clip=True):
    """One-step-ahead rolling predictions on `series`. Non-negative if clip=True."""
    if model is None:
        return np.array([])
    X, _ = make_sequences(series, t_in)
    if len(X) == 0:
        return np.array([])
    model.eval()
    with torch.no_grad():
        z = (series - mu) / sd
        Xz, _ = make_sequences(z, t_in)
        pred_norm = model(torch.tensor(Xz).unsqueeze(-1)).numpy()
        pred = pred_norm * sd + mu
    if clip:
        pred = np.maximum(pred, 0)
    return pred


# ── Metrics ────────────────────────────────────────────────────────────
from scipy.special import gammaln
from scipy.optimize import minimize_scalar


def metrics(true, pred):
    """Full metric dict; returns None when the pair is too short."""
    n = min(len(true), len(pred))
    if n < 2:
        return None
    t_, p_ = true[:n], pred[:n]
    return {
        'rmse': float(np.sqrt(np.mean((t_ - p_) ** 2))),
        'mae': float(np.mean(np.abs(t_ - p_))),
        'bias': float(np.mean(p_ - t_)),
        'corr': float(np.corrcoef(t_, p_)[0, 1]) if t_.std() > 0 else 0.0,
    }


# ── Negative-binomial scoring ─────────────────────────────────────────
def nb_dispersion_mom(series, floor=1e-3):
    """Method-of-moments NB dispersion alpha from a count series
    (parameterization: var = mu + alpha * mu^2). Clipped to `floor`
    so underdispersed series fall back to near-Poisson."""
    s = np.asarray(series, dtype=float)
    mu = s.mean()
    var = s.var(ddof=1) if len(s) > 1 else 0.0
    a = (var - mu) / max(mu ** 2, 1e-12)
    return float(max(a, floor))


def nb_logpmf(y, mu, alpha):
    """Gamma-function extension of the NB pmf (size r = 1/alpha, mean mu),
    valid for real y >= 0 and mu > 0."""
    r = 1.0 / max(alpha, 1e-10)
    p0 = r / (r + mu)
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(p0) + y * np.log1p(-p0))


def nb_nll(true, pred, alpha=None):
    """Mean negative-binomial negative log-likelihood per observation.

    mu is taken from `pred` (clipped to > 0); dispersion `alpha`.
    If alpha is None it is profile-fitted by bounded MLE on the scored series.
    Returns {'nb_nll': float, 'alpha': float} or None when the pair is too short.
    """
    n = min(len(true), len(pred))
    if n < 2:
        return None
    t_ = np.maximum(np.asarray(true[:n], dtype=float), 0.0)
    m_ = np.maximum(np.asarray(pred[:n], dtype=float), 1e-8)
    if alpha is None:
        mom = (t_.var(ddof=1) - t_.mean()) / max(t_.mean() ** 2, 1e-12)
        mom = max(np.clip(mom, 1e-4, 1e4), 1e-4)
        res = minimize_scalar(
            lambda la: -float(np.mean(nb_logpmf(t_, m_, np.exp(la)))),
            bounds=(np.log(1e-5), np.log(1e4)), method='bounded')
        alpha = float(np.exp(res.x))
    nll = -float(np.mean(nb_logpmf(t_, m_, alpha)))
    return {'nb_nll': nll, 'alpha': float(alpha)}


# ── Canonical date-aligned evaluation ───────────────────────────────
# One shared temporal index for every model comparison. Predictions must be
# joined to the clean truth on `week_start` (never by array position), so a
# dropped/duplicated week becomes an explicit failure instead of a silent
# misalignment. The scoring window is defined by dates, independently of any
# model's input-context length (T_IN consumes context, it must not redefine
# which calendar weeks get scored).
def align_to_truth(truth_dates, truth_values, pred_dates, pred_values):
    """Inner-join predictions to truth on `week_start` (one-to-one validated).

    Returns (eval_dates, y_true, y_pred) sorted by week_start. Raises
    ValueError on empty intersection or duplicate weeks.
    """
    t = pd.DataFrame({
        'week_start': pd.to_datetime(np.asarray(truth_dates)),
        'y_true': np.asarray(truth_values, dtype=float),
    })
    p = pd.DataFrame({
        'week_start': pd.to_datetime(np.asarray(pred_dates)),
        'y_pred': np.asarray(pred_values, dtype=float),
    })
    t = t.dropna(subset=['week_start'])
    p = p.dropna(subset=['week_start'])
    if t['week_start'].duplicated().any() or p['week_start'].duplicated().any():
        raise ValueError('align_to_truth: duplicate week_start weeks detected')
    ev = t.merge(p, on='week_start', how='inner', validate='one_to_one')
    if len(ev) == 0:
        raise ValueError('align_to_truth: empty date intersection')
    ev = ev.sort_values('week_start').reset_index(drop=True)
    assert ev['week_start'].is_unique
    return (ev['week_start'].values, ev['y_true'].values, ev['y_pred'].values)


def prediction_dates(series_dates, t_in, n_pred):
    """Calendar weeks each rolling one-step-ahead prediction corresponds to.

    predict() consumes series[i:i+t_in] to forecast series[i+t_in], so
    prediction j scores the week at series_dates[t_in + j].
    """
    d = pd.to_datetime(np.asarray(series_dates)).values
    return d[t_in:t_in + n_pred]


def correction_lag_weeks(cluster, bias_type, bp):
    """Effective forward-looking lag (weeks) of the operational Stage-1 correction.

    `_apply_lag` reconstructs truth week ``t`` from observed week ``t + lag``,
    i.e. it is anti-causal by construction. B1/B3 are pure rescalings (lag 0 —
    split order is irrelevant); B2/combined shift by the prior-mean lag.
    """
    if bias_type in ('B1', 'B3'):
        return 0
    if bias_type in ('B2', 'combined'):
        return max(0, int(prior_lag_weeks(bp[cluster])))
    raise ValueError(f"unknown bias_type {bias_type}")


def correct_full_then_split(biased_full, n_train, cluster, bias_type, bp,
                             mode='operational', draw_params=None, draw=None,
                             ref=None, trim_boundary=True, n_val=None):
    """Stage-1 correct the COMPLETE series first, then split into train/val.

    Correcting train/val/test chunks independently creates artificial
    boundaries: a lag correction at the first validation week cannot see the
    final training weeks it needs (C3 lag: mean ~2wk, max 10wk vs T_IN=8).
    Correcting the full series preserves that cross-boundary history; the
    split below only partitions the already-corrected values.

    Leakage guard (option B): because the lag correction reads ``lag`` weeks
    into the future (corrected[t] = observed[t + lag]), the last ``lag``
    corrected weeks before any split boundary are built from observations on
    the far side of that boundary. With ``trim_boundary=True`` (default) those
    weeks are dropped: train keeps ``corr[:n_train - lag]`` and, when ``n_val``
    is given, val keeps ``corr[n_train:n_train + n_val - lag]``. The test split
    is always returned in full (retrospective scoring inputs). Pass
    ``trim_boundary=False`` to restore the old untrimmed behaviour. B1/B3 have
    lag 0, so trimming is a no-op for them.

    `biased_full` must be the time-ordered train+test concatenation for one
    (cluster, draw). Returns (corrected_train, corrected_rest) when ``n_val``
    is None, else (corrected_train, corrected_val, corrected_test).
    """
    full = np.asarray(biased_full, dtype=float)
    corr = stage1_correct(full, cluster, bias_type, bp, mode=mode,
                          draw_params=draw_params, draw=draw, ref=ref)
    lag = correction_lag_weeks(cluster, bias_type, bp) if trim_boundary else 0
    if lag <= 0:
        if n_val is None:
            return corr[:n_train], corr[n_train:]
        return corr[:n_train], corr[n_train:n_train + n_val], corr[n_train + n_val:]
    n_train_keep = max(0, n_train - lag)
    if n_val is None:
        # Train boundary is clean; the val tail inside `corrected_rest` still
        # needs `n_val` to be trimmed — callers should pass n_val.
        return corr[:n_train_keep], corr[n_train:]
    n_val_keep = max(0, n_val - lag)
    return (corr[:n_train_keep],
            corr[n_train:n_train + n_val_keep],
            corr[n_train + n_val:])


# ── Stage-1 corrections (operational mode only) ────────────────────────
def _prior_mean(p, alpha_col, beta_col):
    a = p[alpha_col]
    b = p[beta_col]
    return a / (a + b + 1e-12)


def prior_rate(p, key):
    """Prior-mean rate for B1 (reporting) or B3 (ascertainment)."""
    if key == 'B1':
        return _prior_mean(p, 'B1_reporting_alpha', 'B1_reporting_beta')
    if key == 'B3':
        return _prior_mean(p, 'B3_detect_alpha', 'B3_detect_beta')
    raise ValueError(key)


def prior_lag_weeks(p):
    """Prior-mean lag (weeks) for B2 from Gamma(shape, scale) in days."""
    return int(round((p['B2_lag_shape'] * p['B2_lag_scale']) / 7.0))


def _apply_lag(series, lag):
    """Shift series: cases appear `lag` weeks late; back-fill tail with last value."""
    n = len(series)
    series = series.astype(float)
    if lag <= 0:
        return series.copy()
    if lag >= n:
        if n == 0:
            return series.copy()
        out = np.full(n, series[-1], dtype=float)
        return out
    out = np.zeros(n)
    out[:n - lag] = series[lag:]
    out[n - lag:] = series[-1]
    return out


def stage1_correct(series, cluster, bias_type, bp, mode='operational',
                   draw_params=None, draw=None, ref=None):
    """Apply the Stage-1 inverse correction to `series`.

    mode: 'operational' only (other modes are no-ops/no-ops for v_05 operational).
    draw_params/draw : ignored in operational mode.
    ref : ignored in operational mode.
    """
    p = bp[cluster]
    s = np.asarray(series, dtype=float)

    if bias_type == 'B1':
        if mode == 'operational':
            rate = prior_rate(p, 'B1')
        else:
            raise ValueError(f"operational arm does not support mode={mode} for B1")
        return s / (rate + 1e-8)

    if bias_type == 'B3':
        if mode == 'operational':
            rate = prior_rate(p, 'B3')
        else:
            raise ValueError(f"operational arm does not support mode={mode} for B3")
        return s / (rate + 1e-8)

    if bias_type == 'B2':
        if mode == 'operational':
            lag = prior_lag_weeks(p)
        else:
            raise ValueError(f"operational arm does not support mode={mode} for B2")
        return _apply_lag(s, lag)

    if bias_type == 'combined':
        # Operational: reverse DGP order (B3 -> B1 -> B2) using prior means only
        if mode == 'operational':
            lag = prior_lag_weeks(p)
            r1 = prior_rate(p, 'B1')
            r3 = prior_rate(p, 'B3')
        else:
            raise ValueError(f"operational arm does not support mode={mode} for combined")
        t = _apply_lag(s, lag)
        t = t / (r1 + 1e-8)
        t = t / (r3 + 1e-8)
        return t

    raise ValueError(f"unknown bias_type {bias_type}")


# ── Adaptive lambda rule (from v_02/03_adaptive_lambda.py) ─────────────
def prior_cv_scale(cluster, bias_type, bp, n_mc=20000, seed=42):
    """CV of the implied multiplicative correction under the cluster prior."""
    p = bp[cluster]
    rng = np.random.default_rng(seed)
    a1, b1 = p['B1_reporting_alpha'], p['B1_reporting_beta']
    a3, b3 = p['B3_detect_alpha'], p['B3_detect_beta']
    k, th = p['B2_lag_shape'], p['B2_lag_scale']
    if bias_type == 'B1':
        s = 1.0 / rng.beta(a1, b1, n_mc)
    elif bias_type == 'B3':
        s = 1.0 / rng.beta(a3, b3, n_mc)
    elif bias_type == 'B2':
        return float(1.0 / np.sqrt(k)), float(k)
    else:  # combined
        s = 1.0 / (rng.beta(a1, b1, n_mc) * rng.beta(a3, b3, n_mc))
    s = np.clip(s, 0, 1e4)
    return float(s.std(ddof=1) / (s.mean() + 1e-12)), None


def lambda_rule(cluster, bias_type, bp):
    """A-priori shrinkage from prior parameters only (no clean labels)."""
    cv, _ = prior_cv_scale(cluster, bias_type, bp)
    KAPPA = 1.0
    lam = 1.0 - KAPPA * cv
    return float(np.clip(lam, 0.0, 1.0))
