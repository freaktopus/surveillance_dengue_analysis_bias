"""
07_model_diagnostics.py
-----------------------
Diagnostics + figures for 06_model_training.py (clean-reference scoring).
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(_SCRIPT_DIR, 'outputs')
OUT  = os.path.join(BASE, '05b_model_training')
FIG  = os.path.join(OUT, 'figures')
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9})

COND_COLORS = {'clean': '#1f77b4', 'B1': '#2ca02c', 'B2': '#ff7f0e',
               'B3': '#d62728', 'combined': '#9467bd'}
COND_ORDER  = ['clean', 'B1', 'B2', 'B3', 'combined']
COND_LEGEND = {'clean': 'clean', 'B1': 'B1 (reporting)',
               'B2': 'B2 (lag)', 'B3': 'B3 (detection)',
               'combined': 'combined'}

# Districts excluded from the study BY CHOICE (kept out of every fallback load).
EXCLUDED_DISTRICTS = ['KASKI']

# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
results = pd.read_csv(os.path.join(OUT, 'model_results.csv'))
print(f"model_results.csv: {len(results)} rows")

# Scoring means (clean 2025, aligned like train_eval). Use predictions CSV when
# available (exact), else fall back to the clean panel.
obs_mean = {}
pred_csv = os.path.join(OUT, 'model_predictions.csv')
if os.path.exists(pred_csv):
    preds = pd.read_csv(pred_csv)
    for c in [1, 2, 3]:
        sub = preds[preds['cluster'] == c]
        obs_mean[c] = float(sub['observed'].mean())
    print(f"model_predictions.csv: {len(preds)} rows (observed means per cluster)")
else:
    preds = None
    weekly = pd.read_csv(os.path.join(BASE, 'panel_weekly.csv'))
    clusters_df = pd.read_csv(os.path.join(BASE, 'district_clusters.csv'))
    weekly = weekly[~weekly['district'].isin(EXCLUDED_DISTRICTS)].copy()
    clusters_df = clusters_df[~clusters_df['district'].isin(EXCLUDED_DISTRICTS)].copy()
    weekly = weekly.merge(clusters_df[['district','cluster']], on='district')
    clean_agg = (weekly.groupby(['cluster','iso_year','week_start'])['weekly_cases'].sum()
                 .reset_index())
    clean_agg['week_start'] = pd.to_datetime(clean_agg['week_start'])
    for c in [1, 2, 3]:
        c25 = clean_agg[(clean_agg['cluster'] == c) & (clean_agg['iso_year'] == 2025)]
        vals = c25['weekly_cases'].values
        obs_mean[c] = vals[8:].mean()   # match te_true alignment (skip T_IN=8)
    print("model_predictions.csv not found; derived scoring means from clean panel.")

# Derive relative RMSE when the CSV predates 05_b's new columns.
if 'rel_rmse' not in results.columns:
    results['rel_rmse'] = results.apply(lambda r: r['rmse'] / obs_mean.get(r['cluster'], 1e-8),
                                        axis=1)
print("rel_rmse derived from rmse + cluster scoring mean (legacy CSV).")

# Centered RMSE (level-removed) per run, computed from the prediction rows:
#   centered_MSE = Var(pred - true) = MSE - bias^2. This is the metric that
#   removes the mean-level advantage; it isolates timing/amplitude skill.
if preds is not None:
    def cen_rmse_of(g):
        d = g['predicted'].values - g['observed'].values
        return float(np.sqrt(np.mean((d - d.mean())**2)))
    cen = (preds.groupby(['cluster','model','condition','draw','seed'])
           .apply(cen_rmse_of).rename('cen_rmse').reset_index())
    results = results.merge(cen, on=['cluster','model','condition','draw','seed'],
                            how='left')
    print("centered RMSE (level-removed) computed from model_predictions.csv")
else:
    print("no model_predictions.csv -> centered RMSE unavailable for this folder")

# Aggregate per cluster x model x condition
agg_dict = {
    'rmse': ('rmse','mean'), 'rel_rmse': ('rel_rmse','mean'),
    'corr': ('corr','mean'), 'bias': ('forecast_bias','mean'),
}
if 'cen_rmse' in results.columns:
    agg_dict['cen_rmse'] = ('cen_rmse','mean')
agg = results.groupby(['cluster','model','condition']).agg(**agg_dict).reset_index()

summary_path = os.path.join(OUT, 'bias_propagation_summary.csv')
summary = pd.read_csv(summary_path) if os.path.exists(summary_path) else None

# ---------------------------------------------------------------------------
# Step 3 — RMSE vs correlation (skill is corr; RMSE gaps = level effects)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
for ax, c in zip(axes, [1, 2, 3]):
    sub = agg[agg['cluster'] == c]
    for cond in COND_ORDER:
        d = sub[sub['condition'] == cond]
        if len(d) == 0:
            continue
        for _, row in d.iterrows():
            mk = 'o' if row['model'] == 'RNN' else 's'
            ax.scatter(row['corr'], row['rmse'], marker=mk, s=70, alpha=.85,
                       color=COND_COLORS[cond], label=cond if row['model'] == 'RNN' else None,
                       zorder=3)
    ax.set_title(f"Cluster {c}")
    ax.set_xlabel("correlation r (scale-free skill)")
    ax.set_xlim(0.5, 0.9)
axes[0].set_ylabel("RMSE (case level)")
# Legend: condition colours + model markers
cond_handles = [plt.Line2D([], [], marker='o', color='none', markerfacecolor=COND_COLORS[c],
                           label=c) for c in COND_ORDER]
model_handles = [plt.Line2D([], [], marker='o', color='none', markerfacecolor='grey',
                            label='RNN'),
                 plt.Line2D([], [], marker='s', color='none', markerfacecolor='grey',
                            label='LSTM')]
handles = cond_handles + model_handles
labels = [h.get_label() for h in handles]
fig.legend(handles, labels, loc='upper center', ncol=len(handles),
           bbox_to_anchor=(0.5, 0.0), frameon=False)
fig.suptitle("Step 3: RMSE vs r per condition (flat r = RMSE gaps are level effects, not skill)",
             y=0.98)
fig.tight_layout(rect=[0, 0.06, 1, 0.95])
fig.savefig(os.path.join(FIG, 'fig_rmse_vs_r.png'), dpi=160)
plt.close(fig)
print("Saved: figures/fig_rmse_vs_r.png")

# ---------------------------------------------------------------------------
# Step 1 — scale-free / level-removed RMSE
#   row 0: rel_rmse  (cross-cluster normaliser; within cluster = const*RMSE)
#   row 1: cen_rmse  (level-removed RMSE = sqrt(MSE - bias^2)) -- the artifact
#                   removal check: does any biased "edge" survive once the
#                   mean-level advantage is factored out?
# ---------------------------------------------------------------------------
rows = [('rel_rmse', 'relative RMSE (RMSE/mean obs)')]
if 'cen_rmse' in agg.columns:
    rows.append(('cen_rmse', 'centered RMSE (level removed)'))

nrow = len(rows)
fig, axes = plt.subplots(nrow, 3, figsize=(13.5, 2.6 * nrow))
if nrow == 1:
    axes = axes.reshape(1, -1)
for i, (metric, mlabel) in enumerate(rows):
    for j, c in enumerate([1, 2, 3]):
        ax = axes[i][j]
        for cond in COND_ORDER:
            d = agg[(agg['cluster'] == c) & (agg['condition'] == cond)]
            if len(d) == 0:
                continue
            vals = d[metric].dropna().values
            if len(vals) == 0:
                continue
            ax.bar(cond, vals.mean(), color=COND_COLORS[cond], alpha=.85,
                   yerr=vals.std() if len(vals) > 1 else None, capsize=3)
        ax.set_title(f"Cluster {c}")
        ax.set_ylabel(mlabel)
        ax.tick_params(axis='x', rotation=30)
# Shared condition legend for the bar panel
bar_handles = [plt.Rectangle((0, 0), 1, 1, color=COND_COLORS[c]) for c in COND_ORDER]
fig.legend(bar_handles, COND_ORDER, loc='lower center', ncol=len(COND_ORDER),
           frameon=False)
fig.suptitle("Step 1: scale-free / level-removed RMSE. "
             "centered RMSE removes the mean-level advantage (bias^2), "
             "isolating timing/amplitude skill.", y=0.99)
fig.tight_layout(rect=[0, 0.05, 1, 0.96])
fig.savefig(os.path.join(FIG, 'fig_scalefree_rmse.png'), dpi=160)
plt.close(fig)
print("Saved: figures/fig_scalefree_rmse.png")

# Print the decomposition table (MSE = bias^2 + cen_rmse^2)
if 'cen_rmse' in agg.columns:
    print("\nStep 1 — decomposition (per cluster x model, RMSE | bias | centered RMSE | corr):")
    base = agg[agg['condition'] == 'clean'].copy()
    for c in [1, 2, 3]:
        for model in ['RNN', 'LSTM']:
            cl = base[(base['cluster'] == c) & (base['model'] == model)]
            if len(cl) == 0:
                continue
            print(f"\n  C{c} {model}: clean RMSE={cl['rmse'].iloc[0]:.1f} "
                  f"cenRMSE={cl['cen_rmse'].iloc[0]:.1f} corr={cl['corr'].iloc[0]:.3f}")
            for cond in ['B1','B2','B3','combined']:
                d = agg[(agg['cluster'] == c) & (agg['model'] == model) & (agg['condition'] == cond)]
                if len(d) == 0:
                    continue
                r0 = d.iloc[0]
                print(f"     {cond:8s} RMSE={r0['rmse']:6.1f}  bias={r0['bias']:+7.1f}  "
                      f"cenRMSE={r0['cen_rmse']:5.1f} ({100*(r0['cen_rmse']/cl['cen_rmse'].iloc[0]-1):+5.1f}% vs clean)  "
                      f"corr={r0['corr']:.3f}")

# ---------------------------------------------------------------------------
# Step 5 — bias magnitude from the data-generating process
# ---------------------------------------------------------------------------
bp = pd.read_csv(os.path.join(BASE, 'bias_parameters.csv'))
rows = []
for _, r in bp.iterrows():
    a1, b1_ = r['B1_reporting_alpha'], r['B1_reporting_beta']
    a3, b3_ = r['B3_detect_alpha'], r['B3_detect_beta']
    b1_rate = a1 / (a1 + b1_)
    b3_probe = a3 / (a3 + b3_)
    combined_retained = b1_rate * b3_probe
    lag_days = r['B2_lag_shape'] * r['B2_lag_scale']
    rows.append({
        'cluster': r['cluster'], 'cluster_label': r['cluster_label'],
        'B1_reporting_rate_mean': round(b1_rate, 3),
        'B3_detection_prob_mean': round(b3_probe, 3),
        'combined_retained': round(combined_retained, 4),
        'combined_attenuation_pct': round((1 - combined_retained) * 100, 1),
        'B2_lag_mean_days': round(lag_days, 1),
        'B2_lag_mean_weeks': round(lag_days / 7, 1),
    })
mag = pd.DataFrame(rows)
mag.to_csv(os.path.join(OUT, 'bias_magnitude_summary.csv'), index=False)
print("\nStep 5 — bias magnitude from the data-generating process:")
print(mag.to_string(index=False))

fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(mag)); w = 0.27
ax.bar(x - w, mag['B1_reporting_rate_mean'], w, label='B1 reporting rate', color='#2ca02c')
ax.bar(x,     mag['B3_detection_prob_mean'], w, label='B3 detection prob',   color='#d62728')
ax.bar(x + w, mag['combined_attenuation_pct'] / 100, w, label='combined attenuation (1-retained)', color='#9467bd')
ax.set_xticks(x); ax.set_xticklabels([f"C{int(r['cluster'])}\n({r['cluster_label']})" for _, r in mag.iterrows()], fontsize=7)
ax.set_ylabel("fraction")
ax.legend()
ax.set_title("Step 5: effective bias magnitude per cluster (Beta/Gamma means)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig_bias_magnitude.png'), dpi=160)
plt.close(fig)
print("Saved: figures/fig_bias_magnitude.png  |  Saved: bias_magnitude_summary.csv")

# ---------------------------------------------------------------------------
# Prediction graphs — REAL (clean 2025) vs MODEL prediction lines
# ---------------------------------------------------------------------------
if preds is not None:
    preds['week_start'] = pd.to_datetime(preds['week_start'])
    for model_name in ['RNN', 'LSTM']:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), sharey=False)
        for j, c in enumerate([1, 2, 3]):
            ax = axes[j]
            sub = preds[(preds['cluster'] == c) & (preds['model'] == model_name)]
            # observed (clean truth) — same for every condition
            obs = (sub[sub['condition'] == 'clean']
                   .groupby('week_start')['observed'].mean())
            ax.plot(obs.index, obs.values, color='black', lw=2.2, label='observed (clean 2025)',
                    zorder=4)
            for cond in COND_ORDER:
                d = sub[sub['condition'] == cond]
                if len(d) == 0:
                    continue
                g = d.groupby('week_start')['predicted']
                mu = g.mean(); sd = g.std(); n = g.size()
                se = sd / np.sqrt(n)
                x_ax = mu.index
                lab = COND_LEGEND.get(cond, cond)
                ax.plot(x_ax, mu.values, color=COND_COLORS[cond], lw=1.6, label=lab)
                ax.fill_between(x_ax, mu.values - 1.96 * se.values,
                                mu.values + 1.96 * se.values,
                                color=COND_COLORS[cond], alpha=.18, lw=0)
            ax.set_title(f"Cluster {c}  ({model_name})")
            ax.set_xlabel("week 2025")
            ax.set_ylabel("cases")
            ax.tick_params(axis='x', rotation=35, labelsize=7)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=6,
                   bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=8)
        fig.suptitle(f"Real vs model predictions — {model_name} "
                     "(lines = mean over draws/seeds; band = 95% CI)", y=0.98)
        fig.tight_layout(rect=[0, 0.07, 1, 0.94])
        fig.savefig(os.path.join(FIG, f'fig_predictions_{model_name}.png'), dpi=160)
        plt.close(fig)
        print(f"Saved: figures/fig_predictions_{model_name}.png")
else:
    print("\nSkipping prediction graphs: model_predictions.csv not found. "
          "Run 06_model_training.py first.")

print("\n✓ Diagnostics complete.")