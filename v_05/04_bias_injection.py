"""
04_bias_injection.py
--------------------
Applies three bias types independently to the clean weekly panel,
using cluster-specific Beta/Gamma distributions for parameter sampling.

B1 — Reporting Rate Bias:
    biased_cases = true_cases * reporting_rate
    reporting_rate ~ Beta(alpha_c, beta_c) per district per draw

B2 — Temporal Lag Bias:
    Cases from week t are recorded in week t + lag
    lag_days ~ Gamma(shape_c, scale_c); converted to weeks

B3 — Ascertainment Bias:
    biased_cases = true_cases * detection_prob
    detection_prob ~ Beta(alpha_c, beta_c) per district per draw

Each bias is applied:
  - Independently (B1 only, B2 only, B3 only)
  - Combined (B1+B2+B3) for filter validation
  - N_DRAWS times to support CI computation

Output (v_05/outputs):
    biased_B1.csv       — reporting rate bias applied
    biased_B2.csv       — temporal lag bias applied
    biased_B3.csv       — ascertainment bias applied
    biased_combined.csv — all three biases combined
    bias_draw_params.csv — sampled parameters per draw/district
"""

import pandas as pd
import numpy as np
import os, warnings

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

N_DRAWS = 100   # Monte Carlo draws per district for CI computation

np.random.seed(42)   # reproducible draws -> identical biased panels across parallel workers

# Districts excluded from the study BY CHOICE (no bias injection, no modelling).
EXCLUDED_DISTRICTS = ['KASKI']

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────
weekly      = pd.read_csv(f'{OUT_DIR}/panel_weekly.csv')
clusters_df = pd.read_csv(f'{OUT_DIR}/district_clusters.csv')
bias_params = pd.read_csv(f'{OUT_DIR}/bias_parameters.csv')

weekly      = weekly[~weekly['district'].isin(EXCLUDED_DISTRICTS)].copy()
clusters_df = clusters_df[~clusters_df['district'].isin(EXCLUDED_DISTRICTS)].copy()

weekly['week_start'] = pd.to_datetime(weekly['week_start'])
weekly = weekly.merge(clusters_df[['district','cluster','cluster_label']], on='district')
weekly = weekly.sort_values(['district','week_start']).reset_index(drop=True)

# Map bias params by cluster
bp = bias_params.set_index('cluster').to_dict('index')
print(f"Bias params: {bp}")

print(f"Weekly panel: {len(weekly)} rows, {weekly['district'].nunique()} districts")
print(f"Monte Carlo draws: {N_DRAWS}\n")

# ── 2. SAMPLE DISTRICT-LEVEL BIAS PARAMETERS ─────────────────────────────────
districts = weekly[['district','cluster']].drop_duplicates()
draw_params_rows = []
for _, drow in districts.iterrows():
    d = drow['district']
    c = drow['cluster']
    p = bp[c]
    for draw in range(N_DRAWS):
        # B1: reporting rate ~ Beta
        b1 = np.random.beta(p['B1_reporting_alpha'], p['B1_reporting_beta'])
        # B2: lag days ~ Gamma → lag weeks (rounded)
        lag_days = np.random.gamma(p['B2_lag_shape'], p['B2_lag_scale'])
        b2_lag_weeks = max(0, round(lag_days / 7))
        # B3: detection probability ~ Beta
        b3 = np.random.beta(p['B3_detect_alpha'], p['B3_detect_beta'])
        draw_params_rows.append({
            'district': d, 'cluster': c, 'draw': draw,
            'B1_reporting_rate': b1,
            'B2_lag_weeks': b2_lag_weeks,
            'B3_detection_prob': b3
        })

draw_params = pd.DataFrame(draw_params_rows)
draw_params.to_csv(f'{OUT_DIR}/bias_draw_params.csv', index=False)
print(f"Sampled {len(draw_params):,} district×draw parameter sets")

# Summary stats
print("\nSampled parameter statistics:")
for c in [1,2,3]:
    sub = draw_params[draw_params['cluster']==c]
    print(f"  Cluster {c}: B1 mean={sub['B1_reporting_rate'].mean():.3f} "
          f"[{sub['B1_reporting_rate'].quantile(0.05):.3f}–{sub['B1_reporting_rate'].quantile(0.95):.3f}] | "
          f"B2 lag={sub['B2_lag_weeks'].mean():.2f}wk | "
          f"B3 det={sub['B3_detection_prob'].mean():.3f} "
          f"[{sub['B3_detection_prob'].quantile(0.05):.3f}–{sub['B3_detection_prob'].quantile(0.95):.3f}]")

# ── 3. BIAS APPLICATION FUNCTIONS ────────────────────────────────────────────

def apply_B1(weekly_df, draw_params_df, draw):
    """Reporting rate bias: multiply cases by district-specific reporting rate."""
    params = draw_params_df[draw_params_df['draw']==draw][['district','B1_reporting_rate']]
    df = weekly_df.merge(params, on='district', how='left')
    df['biased_cases'] = df['weekly_cases'] * df['B1_reporting_rate']
    df['bias_type'] = 'B1_reporting'
    df['draw'] = draw
    return df[['district','cluster','cluster_label','week_start','iso_year','iso_week',
               'weekly_cases','biased_cases','bias_type','draw','B1_reporting_rate']]


def apply_B2(weekly_df, draw_params_df, draw):
    """
    Temporal lag bias: shift each district's time series forward by lag_weeks.
    Cases reported late appear in a later week; missing early weeks → 0.
    """
    params = draw_params_df[draw_params_df['draw']==draw][['district','B2_lag_weeks']]
    result_rows = []
    for district, ddata in weekly_df.groupby('district'):
        lag = params.loc[params['district']==district, 'B2_lag_weeks'].values[0]
        ddata = ddata.sort_values('week_start').reset_index(drop=True)
        n = len(ddata)
        lagged_cases = np.zeros(n)
        if lag == 0:
            lagged_cases = ddata['weekly_cases'].values.copy()
        else:
            lag = int(lag)
            lagged_cases[lag:] = ddata['weekly_cases'].values[:n-lag]
        for i, row in ddata.iterrows():
            r = row.to_dict()
            r['biased_cases'] = lagged_cases[i]
            r['bias_type'] = 'B2_lag'
            r['draw'] = draw
            r['B2_lag_weeks'] = lag
            result_rows.append(r)
    df = pd.DataFrame(result_rows)
    return df[['district','cluster','cluster_label','week_start','iso_year','iso_week',
               'weekly_cases','biased_cases','bias_type','draw','B2_lag_weeks']]


def apply_B3(weekly_df, draw_params_df, draw):
    """Ascertainment bias: multiply cases by district-specific detection probability."""
    params = draw_params_df[draw_params_df['draw']==draw][['district','B3_detection_prob']]
    df = weekly_df.merge(params, on='district', how='left')
    df['biased_cases'] = df['weekly_cases'] * df['B3_detection_prob']
    df['bias_type'] = 'B3_ascertainment'
    df['draw'] = draw
    return df[['district','cluster','cluster_label','week_start','iso_year','iso_week',
               'weekly_cases','biased_cases','bias_type','draw','B3_detection_prob']]


def apply_combined(weekly_df, draw_params_df, draw):
    """
    Combined B1+B2+B3:
    1. Apply ascertainment (B3) → detected cases
    2. Apply reporting rate (B1) → reported cases  
    3. Apply lag (B2) → temporally displaced reported cases
    Order reflects the real data-generating process:
    only detected cases can be reported, and reported cases arrive with lag.
    """
    params = draw_params_df[draw_params_df['draw']==draw][
        ['district','B1_reporting_rate','B2_lag_weeks','B3_detection_prob']]
    result_rows = []
    for district, ddata in weekly_df.groupby('district'):
        p = params[params['district']==district].iloc[0]
        ddata = ddata.sort_values('week_start').reset_index(drop=True)
        n = len(ddata)
        # Step 1: ascertainment
        detected = ddata['weekly_cases'].values * p['B3_detection_prob']
        # Step 2: reporting rate
        reported = detected * p['B1_reporting_rate']
        # Step 3: temporal lag
        lag = int(p['B2_lag_weeks'])
        lagged = np.zeros(n)
        if lag == 0:
            lagged = reported.copy()
        else:
            lagged[lag:] = reported[:n-lag]
        for i, row in ddata.iterrows():
            r = row.to_dict()
            r['biased_cases'] = lagged[i]
            r['bias_type'] = 'combined'
            r['draw'] = draw
            r['B1_reporting_rate'] = p['B1_reporting_rate']
            r['B2_lag_weeks'] = p['B2_lag_weeks']
            r['B3_detection_prob'] = p['B3_detection_prob']
            result_rows.append(r)
    df = pd.DataFrame(result_rows)
    return df[['district','cluster','cluster_label','week_start','iso_year','iso_week',
               'weekly_cases','biased_cases','bias_type','draw',
               'B1_reporting_rate','B2_lag_weeks','B3_detection_prob']]


# ── 4. RUN ALL DRAWS ───────────────────────────────────────────────────────────
print(f"\nApplying biases across {N_DRAWS} draws...")

b1_all, b2_all, b3_all, bc_all = [], [], [], []

for draw in range(N_DRAWS):
    if draw % 25 == 0:
        print(f"  Draw {draw}/{N_DRAWS}...")
    b1_all.append(apply_B1(weekly, draw_params, draw))
    b2_all.append(apply_B2(weekly, draw_params, draw))
    b3_all.append(apply_B3(weekly, draw_params, draw))
    bc_all.append(apply_combined(weekly, draw_params, draw))

biased_B1  = pd.concat(b1_all, ignore_index=True)
biased_B2  = pd.concat(b2_all, ignore_index=True)
biased_B3  = pd.concat(b3_all, ignore_index=True)
biased_comb = pd.concat(bc_all, ignore_index=True)

print("  Done.\n")

# ── 5. SAVE ───────────────────────────────────────────────────────────────────
biased_B1.to_csv(f'{OUT_DIR}/biased_B1.csv', index=False)
biased_B2.to_csv(f'{OUT_DIR}/biased_B2.csv', index=False)
biased_B3.to_csv(f'{OUT_DIR}/biased_B3.csv', index=False)
biased_comb.to_csv(f'{OUT_DIR}/biased_combined.csv', index=False)

print("Saved: biased_B1.csv, biased_B2.csv, biased_B3.csv, biased_combined.csv")

# ── 6. BIAS EFFECT SUMMARY ────────────────────────────────────────────────────
print("\n=== Bias Effect Summary (mean across all draws, cluster-aggregated) ===")
for label, bdf in [('B1',biased_B1),('B2',biased_B2),('B3',biased_B3),('Combined',biased_comb)]:
    agg = bdf.groupby('cluster').agg(
        true_total=('weekly_cases','sum'),
        biased_total=('biased_cases','sum')
    ).reset_index()
    agg['true_total'] /= N_DRAWS
    agg['biased_total'] /= N_DRAWS
    agg['attenuation_pct'] = (1 - agg['biased_total'] / agg['true_total']) * 100
    print(f"\n  {label}:")
    for _, r in agg.iterrows():
        print(f"    Cluster {int(r['cluster'])}: "
              f"true={r['true_total']:.0f} | biased={r['biased_total']:.0f} | "
              f"attenuation={r['attenuation_pct']:.1f}%")

print("\n✓ Bias injection complete.")