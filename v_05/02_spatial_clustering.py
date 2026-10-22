"""
02_spatial_clustering.py
------------------------
K-means clustering of Nepal's 77 districts into 3 surveillance clusters
(C1=Urban/High, C2=Peri-urban/Mid, C3=Rural/Low) using:
  - urbanization_pct
  - health_facility_per100k
  - population_density

Also defines bias injection parameters per cluster (Beta/Gamma priors)
and produces cluster assignment outputs + diagnostic figures.

Output (v_05/outputs):
    district_clusters.csv   — district → cluster assignment + covariates
    bias_parameters.csv     — cluster → Beta/Gamma distribution params per bias

Output (v_05/figures):
    cluster_visualization.png  — 3D scatter of clusters + silhouette/CH/DB score plot
"""

import pandas as pd
import numpy as np
import os
import warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
FIG_DIR = os.path.join(_SCRIPT_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# ── 1. LOAD DISTRICT COVARIATES (produced by 01_data_ingestion) ───────────────
COV_PATH = os.path.join(_SCRIPT_DIR, 'inputs', 'district_covariates.csv')
print(f"Loading district covariates from: {COV_PATH}")
cov_df = pd.read_csv(COV_PATH)

expected_cols = ['district', 'province', 'population', 'area_km2',
                 'urbanization_pct', 'population_density', 'health_facility_per100k']
missing = [c for c in expected_cols if c not in cov_df.columns]
if missing:
    raise ValueError(f"Missing columns in district_covariates.csv: {missing}")

# NOTE: KASKI is RETAINED here. Clustering runs on the full 77-district set so the
# surveillance tiers (C1/C2/C3) keep their meaning; KASKI is excluded only in the
# downstream MODEL TRAINING scripts (see EXCLUDED_DISTRICTS elsewhere).
cov_df['district'] = cov_df['district'].str.strip().str.upper()
assert len(cov_df) == 77, f"Expected 77 districts, got {len(cov_df)}"

# ── 2. BUILD CLUSTERING FEATURES ──────────────────────────────────────────────
FEATURES = ['urbanization_pct', 'health_facility_per100k', 'population_density']
X = cov_df[FEATURES].values.astype(float)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 3. K-MEANS CLUSTERING (k=3) ───────────────────────────────────────────────
print("\nComputing silhouette scores for k=2,3,4,5...")
sil_scores = {}
ch_scores = {}
db_scores = {}

for k in [2, 3, 4, 5]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)
    ch_scores[k] = calinski_harabasz_score(X_scaled, labels)
    db_scores[k] = davies_bouldin_score(X_scaled, labels)
    print(f"k={k}: Silhouette={sil_scores[k]:.4f}, CH={ch_scores[k]:.1f}, DB={db_scores[k]:.4f}")

# Use k=3 as pre-specified (epidemiologically motivated), not necessarily
# the silhouette-maximizing k, to match the v_04/tier-based taxonomy.
best_k = 3
km3 = KMeans(n_clusters=3, random_state=42, n_init=20)
raw_labels = km3.fit_predict(X_scaled)
cov_df['raw_cluster'] = raw_labels + 1  # 1-indexed for compatibility

# ── 4. RELABEL CLUSTERS (1=urban, 2=peri-urban, 3=rural) ─────────────────────
# Map raw cluster → tier by mean urbanization (descending)
cluster_means = cov_df.groupby('raw_cluster')['urbanization_pct'].mean().sort_values(ascending=False)
label_map = {old: new+1 for new, old in enumerate(cluster_means.index)}
cov_df['cluster'] = cov_df['raw_cluster'].map(label_map)

cluster_labels = {1: 'Urban / High Surveillance',
                  2: 'Peri-urban / Mid Surveillance',
                  3: 'Rural / Low Surveillance'}
cov_df['cluster_label'] = cov_df['cluster'].map(cluster_labels)

print("\n=== Cluster Composition ===")
for c in [1, 2, 3]:
    sub = cov_df[cov_df['cluster'] == c]
    print(f"\nCluster {c} — {cluster_labels[c]} (n={len(sub)})")
    print(f"  Urbanization (mean): {sub['urbanization_pct'].mean():.1f}%")
    print(f"  Health fac/100k (mean): {sub['health_facility_per100k'].mean():.2f}")
    print(f"  Pop density (mean): {sub['population_density'].mean():.1f} /km²")
    print(f"  Districts: {', '.join(sorted(sub['district'].tolist()))}")

# ── 5. BIAS PARAMETER DEFINITIONS (Beta/Gamma distributions) ──────────────────
# These are TIER-BASED priors (unchanged from v_01): cluster 1 = high reporting/low lag/high detection,
# cluster 2 = moderate, cluster 3 = low reporting/high lag/low detection.
bias_params = {
    1: {
        'B1_reporting_alpha': 17.0,  'B1_reporting_beta': 3.0,    # mean=0.85, var≈0.007
        'B2_lag_shape':       1.0,   'B2_lag_scale': 1.0,       # mean=1 day
        'B3_detect_alpha':    15.0,  'B3_detect_beta': 5.0,       # mean=0.75, var≈0.009
    },
    2: {
        'B1_reporting_alpha': 6.0,   'B1_reporting_beta': 4.0,    # mean=0.60, var≈0.015
        'B2_lag_shape':       2.5,   'B2_lag_scale': 2.0,       # mean=5 days
        'B3_detect_alpha':    5.0,   'B3_detect_beta': 5.0,       # mean=0.50, var≈0.023
    },
    3: {
        'B1_reporting_alpha': 3.5,   'B1_reporting_beta': 6.5,    # mean=0.35 , var≈0.016
        'B2_lag_shape':       3.5,   'B2_lag_scale': 4.0,       # mean=14 days
        'B3_detect_alpha':    2.5,   'B3_detect_beta': 7.5,       # mean=0.25, var≈0.019
    },
}

print("\n=== Bias Parameter Summary ===")
for c, params in bias_params.items():
    a1, b1 = params['B1_reporting_alpha'], params['B1_reporting_beta']
    a3, b3 = params['B3_detect_alpha'],    params['B3_detect_beta']
    lag_mean = params['B2_lag_shape'] * params['B2_lag_scale']
    print(f"Cluster {c}: B1 mean={a1/(a1+b1):.2f} | B2 lag mean={lag_mean:.1f}d | B3 mean={a3/(a3+b3):.2f}")

# ── 6. SAVE district_clusters.csv ──────────────────────────────────────────────
out_cols = ['district', 'province', 'cluster', 'cluster_label',
            'population', 'area_km2', 'population_density',
            'urbanization_pct', 'health_facility_per100k']
districts_out = cov_df[out_cols].copy()
districts_out_path = os.path.join(OUT_DIR, 'district_clusters.csv')
districts_out.to_csv(districts_out_path, index=False)
print(f"\nSaved: {districts_out_path}")

# ── 7. Save bias_parameters.csv ────────────────────────────────────────────────
bp_rows = []
for c, params in bias_params.items():
    row = {'cluster': c, 'cluster_label': cluster_labels[c]}
    row.update(params)
    bp_rows.append(row)
bp_df = pd.DataFrame(bp_rows)
bp_path = os.path.join(OUT_DIR, 'bias_parameters.csv')
bp_df.to_csv(bp_path, index=False)
print(f"Saved: {bp_path}")

# ── 8. Build cluster_summary.csv ───────────────────────────────────────────────
cluster_summary = cov_df.groupby(['cluster', 'cluster_label']).agg(
    n_districts=('district', 'count'),
    mean_pop_density=('population_density', 'mean'),
    mean_urban_pct=('urbanization_pct', 'mean'),
    mean_health_fac=('health_facility_per100k', 'mean'),
    total_population=('population', 'sum')
).reset_index()
summary_path = os.path.join(FIG_DIR, 'cluster_summary.csv')
cluster_summary.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")

# ── 9. PRODUCE DIAGNOSTIC FIGURES ─────────────────────────────────────────────
# 9a. Silhouette / CH / DB score plot for k=2..5
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

k_vals = [2, 3, 4, 5]
# Silhouette
axes[0].plot(k_vals, [sil_scores[k] for k in k_vals], 'b-o')
axes[0].axvline(3, color='red', linestyle='--', label='k=3 (used)')
axes[0].set_xlabel('k')
axes[0].set_ylabel('Silhouette Score')
axes[0].set_title('Silhouette Score by k')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Calinski-Harabasz
axes[1].plot(k_vals, [ch_scores[k] for k in k_vals], 'r-o')
axes[1].axvline(3, color='red', linestyle='--', label='k=3 (used)')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Calinski-Harabasz Score')
axes[1].set_title('CH Score by k')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Davies-Bouldin
axes[2].plot(k_vals, [db_scores[k] for k in k_vals], 'g-o')
axes[2].axvline(3, color='red', linestyle='--', label='k=3 (used)')
axes[2].set_xlabel('k')
axes[2].set_ylabel('Davies-Bouldin Score')
axes[2].set_title('DB Score by k')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
fig_path = os.path.join(FIG_DIR, 'cluster_scores.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path}")

# 9b. 3D scatter of clusters in feature space
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

scatter_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for i, (cluster_id, color) in enumerate(zip([1, 2, 3], scatter_colors)):
    sub = cov_df[cov_df['cluster'] == cluster_id]
    ax.scatter(sub['population_density'], sub['urbanization_pct'], sub['health_facility_per100k'],
               c=color, label=cluster_labels[cluster_id], s=40, depthshade=True)

ax.set_xlabel('Population Density (/km²)')
ax.set_ylabel('Urbanization (%)')
ax.set_zlabel('Health Facilities /100k')
ax.set_title('District Clusters in Feature Space')
ax.legend()

fig_path2 = os.path.join(FIG_DIR, 'cluster_3d_scatter.png')
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {fig_path2}")

print("\n✓ Spatial clustering complete.")