"""
01_data_ingestion.py
--------------------
Nepal Dengue v_05 Data Ingestion Pipeline
Reads and validates district covariates from v_05/inputs/district_covariates.csv,
produces a clean validated table in v_05/outputs for downstream pipeline scripts.
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# v_05 inputs sit next to the outputs folder (per pipeline design)
INPUTS_DIR = os.path.join(_SCRIPT_DIR, 'inputs')
COV_PATH = os.path.join(INPUTS_DIR, 'district_covariates.csv')

# ── 1. LOAD DISTRICT COVARIATES ──────────────────────────────────────────────
print(f"Loading district covariates from: {COV_PATH}")
cov_df = pd.read_csv(COV_PATH)

expected_cols = ['district', 'province', 'population', 'area_km2',
                 'urbanization_pct', 'population_density', 'health_facility_per100k']
missing = [c for c in expected_cols if c not in cov_df.columns]
if missing:
    raise ValueError(f"Missing columns in district_covariates.csv: {missing}")

extra = [c for c in cov_df.columns if c not in expected_cols]
if extra:
    print(f"Warning: Extra columns ignored: {extra}")

cov_df = cov_df[expected_cols].copy()

# ── 2. VALIDATE DISTRICT LIST ────────────────────────────────────────────────
CANONICAL_DISTRICTS = [
    'TAPLEJUNG','SANKHUWASABHA','SOLUKHUMBU','OKHALDHUNGA','KHOTANG',
    'BHOJPUR','DHANKUTA','TERHATHUM','PANCHTHAR','ILAM','JHAPA','MORANG',
    'SUNSARI','UDAYAPUR','SAPTARI','SIRAHA','DHANUSA','MAHOTTARI','SARLAHI',
    'RAUTAHAT','BARA','PARSA','DOLAKHA','SINDHUPALCHOK','RASUWA','DHADING',
    'NUWAKOT','KATHMANDU','BHAKTAPUR','LALITPUR','KAVREPALANCHOK','RAMECHHAP',
    'SINDHULI','MAKWANPUR','CHITAWAN','GORKHA','MANANG','MUSTANG','MYAGDI',
    'KASKI','LAMJUNG','TANAHU','NAWALPARASI EAST','NAWALPARASI WEST','SYANGJA',
    'PARBAT','BAGLUNG','RUKUM EAST','ROLPA','PYUTHAN','GULMI','ARGHAKHANCHI',
    'PALPA','RUPANDEHI','KAPILBASTU','DANG','BANKE','BARDIYA','DOLPA','MUGU',
    'HUMLA','JUMLA','KALIKOT','DAILEKH','JAJARKOT','RUKUM WEST','SALYAN',
    'SURKHET','BAJURA','BAJHANG','DARCHULA','BAITADI','DADELDHURA','DOTI',
    'ACHHAM','KAILALI','KANCHANPUR'
]

assert len(CANONICAL_DISTRICTS) == 77, "District count mismatch"
cov_df['district'] = cov_df['district'].str.strip().str.upper()
non_canonical = set(cov_df['district'].unique()) - set(CANONICAL_DISTRICTS)
if non_canonical:
    raise ValueError(f"Non-canonical districts found: {non_canonical}")

missing_dists = set(CANONICAL_DISTRICTS) - set(cov_df['district'].unique())
if missing_dists:
    raise ValueError(f"Missing canonical districts: {missing_dists}")

print(f"Districts validated: {len(cov_df)} / 77 canonical districts")

# ── 3. SAVE VALIDATED TABLE ──────────────────────────────────────────────────
out_path = os.path.join(OUT_DIR, 'district_covariates.csv')
cov_df.to_csv(out_path, index=False)
print(f"Saved validated covariates to: {out_path}")

# ── 4. BASIC SUMMARY ─────────────────────────────────────────────────────────
summary = f"""v_05 Data Ingestion Report
==========================
Input file    : {COV_PATH}
Districts     : {len(cov_df)} canonical districts
Columns       : {list(cov_df.columns)}
Province      : {sorted(cov_df['province'].unique())}
Urbanization  : mean={cov_df['urbanization_pct'].mean():.2f}, 
                min={cov_df['urbanization_pct'].min():.2f}, 
                max={cov_df['urbanization_pct'].max():.2f}
Population density: mean={cov_df['population_density'].mean():.2f}, 
                min={cov_df['population_density'].min():.2f},
                max={cov_df['population_density'].max():.2f}
Health fac/100k: mean={cov_df['health_facility_per100k'].mean():.2f},
              min={cov_df['health_facility_per100k'].min():.2f},
              max={cov_df['health_facility_per100k'].max():.2f}
"""

print(summary)
with open(os.path.join(OUT_DIR, 'ingestion_report.txt'), 'w', encoding='utf-8') as f:
    f.write(summary)

print("\n✓ Data ingestion complete.")