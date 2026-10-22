"""
08_sier_recalibrated.py
-----------------------
Seasonally-forced SEIR with rolling-window refit (recalibration) for dengue
incidence by spatial cluster.

Model (weekly, discrete-time):
    S[t+1] = S[t] - beta(t)*S[t]*I[t]/N
    E[t+1] = E[t] + beta(t)*S[t]*I[t]/N - sigma*E[t]
    I[t+1] = I[t] + sigma*E[t] - gamma*I[t]
    R[t+1] = R[t] + gamma*I[t]

    beta(t) = beta0 * (1 + A * exp(-(iso_week - mu)^2 / (2*sigma_p^2)))
    mu      = transmission peak ISO week (fit in [28,38]: Aug-Sep, so the
              incidence peak lags to ~ISO week 43, matching observations)
    sigma   = 1/1   (1-week latent period)
    gamma   = 1/2   (2-week infectious period)

Free parameters (Nelder-Mead + multiple restarts):
    beta0, A, sigma_p, mu, E0_frac, I0_frac

Fitting losses (two parallel setups, single script):
    rmse  — sqrt-scale MSE: mean((sqrt(pred+1)-sqrt(obs+1))**2) [legacy behaviour]
    nbnll — mean negative-binomial NLL: -mean(log p_NB(y; mu=pred, alpha)),
            var = mu + alpha*mu^2, alpha fixed per fit window by
            method-of-moments (leakage-free: estimated from the training
            window only, never from the forecast horizon).

Validation:
    - Calibration: all of 2022-2024 (in-sample).
    - Rolling refit: initial fit on 2022-2024, then every k=4 weeks re-fit on a
      trailing 26-week window and forecast the next k weeks out-of-sample.
    - Diagnostics: strict no-refit 2025 projection (peak timing; shows why the
      rolling refit is necessary).



Run:
    python 08_sier_recalibrated.py                      # both arms (default)
    python 08_sier_recalibrated.py --losses rmse        # rmse arm only
    python 08_sier_recalibrated.py --losses nbnll       # nbnll arm only
    python 08_sier_recalibrated.py --clusters 1 2 --n-restarts-cal 3 --n-restarts-rolling 2 --seed 42  # smoke test
"""

import argparse
import os
import shutil
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(_SCRIPT_DIR, 'outputs')
OUT_DIR = os.path.join(BASE, '05b_model_training')
os.makedirs(OUT_DIR, exist_ok=True)

# Calibration window: 2022-2024; holdout: 2025
TRAIN_YEARS = [2022, 2023, 2024]
TEST_YEAR = 2025

# Fixed dengue biology (weekly units)
SIGMA = 1.0 / 1.0   # 1-week latent
GAMMA = 1.0 / 2.0   # 2-week infectious

# Rolling-refit protocol (defaults; overridable via CLI for smoke tests)
HORIZON_K = 4       # forecast horizon (weeks)
WINDOW_LEN = 26     # trailing window length
N_RESTARTS_CAL = 30
N_RESTARTS_ROLLING = 16

CLUSTER_LABELS = {
    1: 'Urban / High Surveillance',
    2: 'Peri-urban / Mid Surveillance',
    3: 'Rural / Low Surveillance',
}

# Districts excluded from the study BY CHOICE (also removed from the SEIR
# population denominator N).
EXCLUDED_DISTRICTS = ['KASKI']

MU_EPS = 1e-8
ALPHA_FLOOR = 1e-3


# ── NB helpers (leakage-free MoM; correct -mean(logpmf) sign) ────────────────
def nb_dispersion_mom(series, floor=ALPHA_FLOOR):
    """MoM NB dispersion alpha: var = mu + alpha*mu^2. Near-Poisson fallback."""
    s = np.asarray(series, dtype=float)
    if len(s) < 2:
        return float(floor)
    mu = s.mean()
    var = s.var(ddof=1)
    a = (var - mu) / max(mu ** 2, 1e-12)
    return float(max(a, floor))


def nb_logpmf_vec(y, mu, alpha):
    """Gamma-extension NB log-pmf, vectorised. y >= 0 real, mu > 0."""
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), MU_EPS)
    r = 1.0 / max(float(alpha), 1e-10)
    p0 = r / (r + mu)
    # log1p(-p0) = log(mu/(r+mu)) with better precision via log(r+mu) form
    return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
            + r * np.log(p0) + y * (np.log(mu) - np.log(r + mu)))


def nb_mean_nll(observed, predicted, alpha):
    """Mean NB negative log-likelihood per observation (lower = better)."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if len(obs) < 1 or len(pred) < 1:
        return float('nan')
    n = min(len(obs), len(pred))
    ll = nb_logpmf_vec(np.maximum(obs[:n], 0.0), np.maximum(pred[:n], MU_EPS), alpha)
    if not np.all(np.isfinite(ll)):
        return float('inf')
    return float(-np.mean(ll))


# ── SEASONAL SEIR MODEL ───────────────────────────────────────────────────
def seasonal_beta(iso_week, beta0, A, sigma_p, mu):
    """Calendar-anchored transmission rate (Gaussian pulse centred on mu)."""
    d = np.abs(np.asarray(iso_week) - mu)
    d = np.minimum(d, 52 - d)  # wrap around the year boundary
    return beta0 * (1 + A * np.exp(-d**2 / (2 * sigma_p**2)))


def seir_simulate(beta0, A, sigma_p, mu, E0, I0, N, iso_weeks):
    """Discrete-time seasonal SEIR. Returns new_infected (sigma*E) per week."""
    S = N - E0 - I0
    E, I, R = E0, I0, 0.0
    new_cases = []
    for w in iso_weeks:
        beta = seasonal_beta(w, beta0, A, sigma_p, mu)
        new_exposed = beta * S * I / N
        new_infected = SIGMA * E
        new_removed = GAMMA * I
        S = max(S - new_exposed, 0)
        E = max(E + new_exposed - new_infected, 0)
        I = max(I + new_infected - new_removed, 0)
        R = R + new_removed
        new_cases.append(new_infected)
    return np.array(new_cases)


def fit_seir_seasonal(observed, iso_weeks, N, n_restarts, loss='rmse', alpha=None):
    """Fit seasonal SEIR with multiple Nelder-Mead restarts.

    loss='rmse'  : sqrt-scale MSE  mean((sqrt(pred+1)-sqrt(obs+1))**2)
    loss='nbnll' : mean NB-NLL with FIXED alpha (MoM from the fit window).
    """
    observed = np.asarray(observed, dtype=float)
    iso_weeks = np.asarray(iso_weeks)
    if loss == 'nbnll' and alpha is None:
        raise ValueError('nbnll fit requires a fixed alpha (MoM from fit window)')

    best_loss = np.inf
    best_params = None

    for _ in range(n_restarts):
        beta0_0 = np.random.uniform(0.02, 0.3)
        A_0 = np.random.uniform(0.2, 10.0)
        sp_0 = np.random.uniform(4.0, 10.0)
        mu_0 = np.random.uniform(29.0, 36.0)
        E0f_0 = np.random.uniform(1e-5, 1e-3)
        I0f_0 = np.random.uniform(1e-5, 1e-3)

        def loss_fn(params):
            beta0, A, sigma_p, mu, E0f, I0f = params
            base_r0 = beta0 / GAMMA
            peak_r0 = beta0 * (1 + A) / GAMMA
            if (beta0 <= 0 or A < 0.05 or A > 100 or sigma_p < 3 or sigma_p > 14
                    or mu < 28 or mu > 38 or E0f <= 0 or I0f <= 0
                    or E0f > 0.01 or I0f > 0.01
                    or base_r0 > 0.8 or peak_r0 > 5.0):
                return 1e12
            E0, I0 = E0f * N, I0f * N
            if N - E0 - I0 <= 0:
                return 1e12
            try:
                pred = seir_simulate(beta0, A, sigma_p, mu, E0, I0, N, iso_weeks)
                if loss == 'nbnll':
                    nll = nb_mean_nll(observed, pred, alpha)
                    return nll if np.isfinite(nll) else 1e12
                return np.mean((np.sqrt(pred + 1) - np.sqrt(observed + 1))**2)
            except Exception:
                return 1e12

        res = minimize(loss_fn, [beta0_0, A_0, sp_0, mu_0, E0f_0, I0f_0],
                       method='Nelder-Mead',
                       options={'maxiter': 1200, 'xatol': 1e-8, 'fatol': 1e-8})

        if res.fun < best_loss:
            best_loss = res.fun
            best_params = res.x

    beta0, A, sigma_p, mu, E0f, I0f = best_params
    E0 = int(E0f * N)
    I0 = int(I0f * N)
    S0 = int(N - E0 - I0)
    fitted = seir_simulate(beta0, A, sigma_p, mu, E0, I0, N, iso_weeks)

    return {
        'beta0': beta0, 'A': A, 'sigma_p': sigma_p, 'mu': mu,
        'E0_frac': E0f, 'I0_frac': I0f,
        'E0': E0, 'I0': I0, 'S0': S0,
        'peak_R0': beta0 * (1 + A) / GAMMA,
        'base_R0': beta0 / GAMMA,
        'fit_loss': float(best_loss),
        'loss': loss,
        'alpha_used': float(alpha) if loss == 'nbnll' else np.nan,
        'nll': float(best_loss),  # legacy alias (fit loss, whatever the loss is)
        'fitted': fitted,
    }


# ── METRICS (side-by-side Gaussian + count) ──────────────────────────────────
def metrics(observed, predicted):
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    ss_res = np.sum((observed - predicted)**2)
    ss_tot = np.sum((observed - observed.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    rmse = np.sqrt(np.mean((observed - predicted)**2))
    mae = np.mean(np.abs(observed - predicted))
    return r2, rmse, mae


def load_cluster_weekly():
    weekly = pd.read_csv(f'{BASE}/panel_weekly.csv')
    clusters_df = pd.read_csv(f'{BASE}/district_clusters.csv')
    weekly = weekly[~weekly['district'].isin(EXCLUDED_DISTRICTS)].copy()
    clusters_df = clusters_df[~clusters_df['district'].isin(EXCLUDED_DISTRICTS)].copy()
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    weekly = weekly.merge(clusters_df[['district', 'cluster', 'cluster_label', 'population']],
                          on='district', how='left')
    cluster_weekly = (weekly
        .groupby(['cluster', 'cluster_label', 'iso_year', 'iso_week', 'week_start'])
        .agg(weekly_cases=('weekly_cases', 'sum'),
             N=('population', 'sum'))
        .reset_index()
        .sort_values(['cluster', 'week_start'])
        .reset_index(drop=True))
    return cluster_weekly


def run_arm(cluster_weekly, loss, clusters, n_restarts_cal, n_restarts_rolling,
            horizon_k=HORIZON_K, window_len=WINDOW_LEN, seed=0):
    """Run one fitting-loss arm. Returns (params_rows, fitted_rows, metrics_rows)."""
    # Same initialisations for both arms -> fair loss comparison, reproducible.
    np.random.seed(seed)
    print(f"=== Loss arm: {loss} (seed={seed}) ===\n")

    params_rows, fitted_rows, metrics_rows = [], [], []

    for c in clusters:
        cdata = cluster_weekly[cluster_weekly['cluster'] == c].copy()
        N = int(cdata['N'].iloc[0])
        label = CLUSTER_LABELS[c]

        full = cdata.sort_values('week_start').reset_index(drop=True)
        # Canonical calendar masks: filter observations by DATE, never by
        # position. `obs_all[:n_cal]` is wrong whenever the calibration years
        # are not an exact prefix of the array (here the 203-week array starts
        # 2021-12-27, so a positional slice includes a pre-2022 week and drops
        # 2024-12-02) — the model would calibrate against weeks the code does
        # not think they are.
        cal_mask = full['iso_year'].isin(TRAIN_YEARS).values
        test_mask = (full['iso_year'] == TEST_YEAR).values
        cal_idx = np.where(cal_mask)[0]
        test_idx = np.where(test_mask)[0]
        if len(cal_idx) == 0 or len(test_idx) == 0:
            print(f"  Cluster {c}: empty calibration or holdout window, skipping.")
            continue
        n_cal = len(cal_idx)
        first_test = int(test_idx[0])

        obs_all = full['weekly_cases'].values.astype(float)
        iw_all = full['iso_week'].values.astype(int)
        ws_all = full['week_start'].values
        iy_all = full['iso_year'].values.astype(int)
        n_total = len(full)

        obs_cal = obs_all[cal_mask]
        iw_cal = iw_all[cal_mask]

        # Leakage-free eval dispersion: MoM alpha from 2022-2024 calibration only.
        alpha_eval = nb_dispersion_mom(obs_cal)

        print(f"Cluster {c} — {label}")
        print(f"  N={N:,} | Calibration weeks: {n_cal} | Holdout weeks: {len(test_idx)} | alpha_eval={alpha_eval:.4f}")

        # ---- Calibration fit on 2022-2024 ----
        alpha_cal = alpha_eval if loss == 'nbnll' else None
        cal_result = fit_seir_seasonal(obs_cal, iw_cal, N, n_restarts_cal,
                                       loss=loss, alpha=alpha_cal)
        cal_r2, cal_rmse, cal_mae = metrics(obs_cal, cal_result['fitted'])
        cal_nbnll = nb_mean_nll(obs_cal, cal_result['fitted'], alpha_eval)
        print(f"  Calibration: β0={cal_result['beta0']:.4f}, A={cal_result['A']:.2f}, "
              f"σp={cal_result['sigma_p']:.1f}, μ={cal_result['mu']:.0f}, "
              f"base R0={cal_result['base_R0']:.2f}, peak R0={cal_result['peak_R0']:.2f}")
        print(f"    R²={cal_r2:.4f}, RMSE={cal_rmse:.2f}, MAE={cal_mae:.2f}, NB-NLL={cal_nbnll:.4f} "
              f"(fit_loss[{loss}]={cal_result['fit_loss']:.4f})")

        # ---- Strict no-refit 2025 projection (diagnostic) ----
        # Holdout boundary is the first 2025 week by DATE (not n_cal offset).
        strict_pred = seir_simulate(cal_result['beta0'], cal_result['A'], cal_result['sigma_p'],
                                    cal_result['mu'], cal_result['E0'], cal_result['I0'], N, iw_all)
        pred_2025 = strict_pred[test_mask]
        obs_2025 = obs_all[test_mask]
        iw_2025 = iw_all[test_mask]
        obs_peak_w = iw_2025[int(np.argmax(obs_2025))]
        pred_peak_w = iw_2025[int(np.argmax(pred_2025))]
        peak_week_err = pred_peak_w - obs_peak_w
        peak_mag_ratio = pred_2025.max() / obs_2025.max() if obs_2025.max() > 0 else np.nan
        print(f"  Strict no-refit projection: peak week pred={pred_peak_w} obs={obs_peak_w} "
              f"(err {peak_week_err:+d}) — magnitude not extrapolatable across the 2025 regime shift "
              f"(ratio {peak_mag_ratio:.1f}); rolling refit below addresses this.")

        # ---- Rolling refit for 2025 ----
        # Forecasts start at the first 2025 week by DATE. Each rolling fit uses
        # a trailing window over the true date-ordered array (no positional
        # assumption about where the calibration period sits).
        roll = []  # (lead, week_start, iso_year, iso_week, observed, forecast)
        refit_steps = []

        t = first_test  # first index to forecast (first 2025 week)
        while t < n_total:
            train_end = t
            train_start = max(0, train_end - window_len)
            obs_tr = obs_all[train_start:train_end]
            iw_tr = iw_all[train_start:train_end]

            alpha_roll = nb_dispersion_mom(obs_tr) if loss == 'nbnll' else None
            res = fit_seir_seasonal(obs_tr, iw_tr, N, n_restarts_rolling,
                                    loss=loss, alpha=alpha_roll)
            refit_steps.append({
                'cluster': c, 'cluster_label': label, 'N': N, 'loss': loss,
                'fit_end_week': ws_all[train_end - 1],
                'n_train_weeks': len(obs_tr),
                'beta0': res['beta0'], 'A': res['A'], 'sigma_p': res['sigma_p'],
                'mu': res['mu'], 'E0_frac': res['E0_frac'], 'I0_frac': res['I0_frac'],
                'E0': res['E0'], 'I0': res['I0'], 'S0': res['S0'],
                'peak_R0': res['peak_R0'], 'base_R0': res['base_R0'],
                'alpha_used': res['alpha_used'], 'fit_loss': res['fit_loss'],
                'nll': res['nll'],
            })

            horizon = min(horizon_k, n_total - train_end)
            iw_sim = iw_all[train_start:train_end + horizon]
            pred_sim = seir_simulate(res['beta0'], res['A'], res['sigma_p'],
                                     res['mu'], res['E0'], res['I0'], N, iw_sim)
            fc = pred_sim[len(obs_tr):len(obs_tr) + horizon]

            for h in range(horizon):
                idx = train_end + h
                roll.append((h + 1, ws_all[idx], iy_all[idx], iw_all[idx], obs_all[idx], fc[h]))

            t += horizon_k

        roll_df = pd.DataFrame(roll, columns=['lead', 'week_start', 'iso_year', 'iso_week', 'observed', 'forecast'])

        # ---- Rolling metrics per horizon (RMSE-family + NB-NLL side by side) ----
        r2_all, rmse_all, mae_all = metrics(roll_df['observed'].values, roll_df['forecast'].values)
        nbnll_all = nb_mean_nll(roll_df['observed'].values, roll_df['forecast'].values, alpha_eval)
        per_lead, per_lead_nbnll = {}, {}
        for lead in sorted(roll_df['lead'].unique()):
            sub = roll_df[roll_df['lead'] == lead]
            per_lead[lead] = metrics(sub['observed'].values, sub['forecast'].values)
            per_lead_nbnll[lead] = nb_mean_nll(sub['observed'].values, sub['forecast'].values, alpha_eval)

        print(f"  Rolling holdout ({len(roll_df)} forecasts): R²={r2_all:.4f}, RMSE={rmse_all:.2f}, "
              f"MAE={mae_all:.2f}, NB-NLL={nbnll_all:.4f}")
        for lead in sorted(per_lead):
            r2_l, rmse_l, mae_l = per_lead[lead]
            print(f"    lead {lead}wk: R²={r2_l:.4f}, RMSE={rmse_l:.2f}, MAE={mae_l:.2f}, "
                  f"NB-NLL={per_lead_nbnll[lead]:.4f}")

        metrics_rows.append({
            'cluster': c, 'cluster_label': label, 'N': N, 'loss': loss,
            'alpha_eval': alpha_eval, 'fit_loss_cal': cal_result['fit_loss'],
            'cal_r2': cal_r2, 'cal_rmse': cal_rmse, 'cal_mae': cal_mae, 'cal_nbnll': cal_nbnll,
            'holdout_r2_all': r2_all, 'holdout_rmse_all': rmse_all,
            'holdout_mae_all': mae_all, 'holdout_nbnll_all': nbnll_all,
            'holdout_r2_lead1': per_lead.get(1, (np.nan,))[0],
            'holdout_r2_lead2': per_lead.get(2, (np.nan,))[0],
            'holdout_r2_lead4': per_lead.get(4, (np.nan,))[0],
            'holdout_rmse_lead1': per_lead.get(1, (np.nan,))[1],
            'holdout_rmse_lead2': per_lead.get(2, (np.nan,))[1],
            'holdout_rmse_lead4': per_lead.get(4, (np.nan,))[1],
            'holdout_nbnll_lead1': per_lead_nbnll.get(1, np.nan),
            'holdout_nbnll_lead2': per_lead_nbnll.get(2, np.nan),
            'holdout_nbnll_lead4': per_lead_nbnll.get(4, np.nan),
            'peak_obs_iso_week': obs_peak_w, 'peak_pred_iso_week': pred_peak_w,
            'peak_week_err': peak_week_err, 'peak_mag_ratio': peak_mag_ratio,
            'n_forecasts': len(roll_df),
        })

        # ---- Store fitted series (calibration rows addressed by DATE mask) ----
        for i in cal_idx:
            fitted_rows.append({
                'cluster': c, 'cluster_label': label, 'loss': loss,
                'week_start': ws_all[i], 'iso_year': int(iy_all[i]),
                'iso_week': int(iw_all[i]), 'observed': float(obs_all[i]),
                'fitted': float(cal_result['fitted'][int(np.where(cal_idx == i)[0][0])]), 'split': 'calibration',
            })
        for _, r in roll_df.iterrows():
            fitted_rows.append({
                'cluster': c, 'cluster_label': label, 'loss': loss,
                'week_start': r['week_start'], 'iso_year': int(r['iso_year']),
                'iso_week': int(r['iso_week']), 'observed': r['observed'],
                'fitted': r['forecast'], 'split': 'rolling_forecast',
            })

        params_rows.extend(refit_steps)
        print()

    return params_rows, fitted_rows, metrics_rows


def save_arm(loss, params_rows, fitted_rows, metrics_rows):
    prefix = f'seir_recalibrated_{loss}'
    pd.DataFrame(params_rows).to_csv(f'{OUT_DIR}/{prefix}_parameters.csv', index=False)
    pd.DataFrame(fitted_rows).to_csv(f'{OUT_DIR}/{prefix}_fitted.csv', index=False)
    pd.DataFrame(metrics_rows).to_csv(f'{OUT_DIR}/{prefix}_metrics.csv', index=False)
    print(f"Saved: {prefix}_parameters.csv, {prefix}_fitted.csv, {prefix}_metrics.csv")


def main():
    ap = argparse.ArgumentParser(description='Seasonal SEIR recalibration: RMSE vs NB-NLL parallel arms')
    ap.add_argument('--losses', nargs='+', choices=['rmse', 'nbnll'], default=['rmse', 'nbnll'],
                    help='fitting-loss arms to run (default: both, non-overlapping outputs)')
    ap.add_argument('--clusters', nargs='+', type=int, default=[1, 2, 3])
    ap.add_argument('--n-restarts-cal', type=int, default=N_RESTARTS_CAL)
    ap.add_argument('--n-restarts-rolling', type=int, default=N_RESTARTS_ROLLING)
    ap.add_argument('--seed', type=int, default=0,
                    help='base RNG seed; reset per arm so both arms share initialisations')
    ap.add_argument('--no-legacy', action='store_true',
                    help='do not copy rmse arm to legacy seir_recalibrated_*.csv names')
    args = ap.parse_args()

    print(f"Seasonal SEIR recalibration (rolling refit, k={HORIZON_K}wk, window={WINDOW_LEN}wk)...")
    print(f"Loss arms: {args.losses} | clusters: {args.clusters} | "
          f"restarts cal/rolling: {args.n_restarts_cal}/{args.n_restarts_rolling}\n")

    cluster_weekly = load_cluster_weekly()

    for loss in args.losses:
        params_rows, fitted_rows, metrics_rows = run_arm(
            cluster_weekly, loss, args.clusters,
            args.n_restarts_cal, args.n_restarts_rolling, seed=args.seed)
        save_arm(loss, params_rows, fitted_rows, metrics_rows)

    # Backward compat: legacy names == rmse arm (only when rmse was run).
    if 'rmse' in args.losses and not args.no_legacy:
        for suffix in ('parameters', 'fitted', 'metrics'):
            shutil.copy(f'{OUT_DIR}/seir_recalibrated_rmse_{suffix}.csv',
                        f'{OUT_DIR}/seir_recalibrated_{suffix}.csv')
        print("Legacy aliases refreshed from rmse arm: seir_recalibrated_{parameters,fitted,metrics}.csv")

    print("\n✓ Seasonal SEIR recalibration complete.")


if __name__ == '__main__':
    main()
