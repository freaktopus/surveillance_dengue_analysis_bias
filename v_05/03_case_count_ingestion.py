"""
03_case_count_ingestion.py
---------------------------
Nepal Dengue v_05 Case Count Ingestion Pipeline
Reads case snapshot CSVs from repo/data/ (2022-2025), derives incident cases,
produces weekly panel. Outputs pushed to root output directory (v_05/outputs)
for the entire pipeline to run.
"""

import pandas as pd
import numpy as np
import glob, os, warnings
from datetime import datetime
import math

warnings.filterwarnings('ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR   = os.path.join(_REPO_DIR, 'data')
OUT_DIR    = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. CANONICAL DISTRICT LIST ────────────────────────────────────────────────
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

# Districts excluded from the study BY CHOICE (no panel, no modelling).
EXCLUDED_DISTRICTS = ['KASKI']
ACTIVE_DISTRICTS = [d for d in CANONICAL_DISTRICTS if d not in EXCLUDED_DISTRICTS]

# ── 2. SCHEMA NORMALIZERS ────────────────────────────────────────────────────
MONTH_ORDER = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

def normalize_district(name):
    if pd.isna(name):
        return None
    n = str(name).strip().upper()
    aliases = {
        'ALL':         '__DROP__',
        'TOTAL':       '__DROP__',
        'GRAND TOTAL': '__DROP__',
    }
    return aliases.get(n, n)

def parse_simple(df, date):
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={'DISTRICT': 'district', 'TOTAL': 'cumulative'})
    df['district'] = df['district'].apply(normalize_district)
    df = df[~df['district'].isin(['__DROP__', None])]
    df['date'] = date
    df['year'] = date.year
    return df[['date','year','district','cumulative']]

def parse_wide(df, date):
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={'DISTRICT': 'district', 'TOTAL': 'cumulative'})
    df['district'] = df['district'].apply(normalize_district)
    df = df[~df['district'].isin(['__DROP__', None])]
    df['date'] = date
    df['year'] = date.year
    MONTH_ORDER = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    month_cols_present = [c for c in df.columns
                          if any(c.startswith(m) for m in MONTH_ORDER)]
    return df[['date','year','district','cumulative']], month_cols_present

# ── 3. LOAD ALL FILES ────────────────────────────────────────────────────────
files = sorted(glob.glob(f'{DATA_DIR}/*.csv'))
print(f"Found {len(files)} files: {os.path.basename(files[0])} → {os.path.basename(files[-1])}")

records = []
skipped = []
schema_counts = {'simple': 0, 'wide': 0}

for fpath in files:
    fname = os.path.basename(fpath)
    try:
        date = pd.to_datetime(fname.replace('.csv',''))
    except Exception:
        skipped.append((fname, 'unparseable date'))
        continue

    try:
        raw = pd.read_csv(fpath)
    except Exception as e:
        skipped.append((fname, f'read error: {e}'))
        continue

    cols_upper = [c.strip().upper() for c in raw.columns]

    if 'PROVINCE' in cols_upper:
        parsed, _ = parse_wide(raw, date)
        schema_counts['wide'] += 1
    else:
        parsed = parse_simple(raw, date)
        schema_counts['simple'] += 1

    records.append(parsed)

panel_cum = pd.concat(records, ignore_index=True)
panel_cum['cumulative'] = pd.to_numeric(panel_cum['cumulative'], errors='coerce').fillna(0).astype(int)

print(f"\nSchema counts: {schema_counts}")
print(f"Panel shape (raw): {panel_cum.shape}")
print(f"Unique districts (raw): {panel_cum['district'].nunique()}")
print(f"Date range: {panel_cum['date'].min().date()} → {panel_cum['date'].max().date()}")

# ── 4. RESTRICT TO CANONICAL DISTRICTS ──────────────────────────────────────
non_canonical = set(panel_cum['district'].unique()) - set(CANONICAL_DISTRICTS)
if non_canonical:
    print(f"\nDropping non-canonical entries: {non_canonical}")

panel_cum = panel_cum[panel_cum['district'].isin(CANONICAL_DISTRICTS)].copy()
print(f"Panel shape (canonical): {panel_cum.shape}")

n_before = panel_cum['district'].nunique()
panel_cum = panel_cum[~panel_cum['district'].isin(EXCLUDED_DISTRICTS)].copy()
print(f"Excluded by choice: {EXCLUDED_DISTRICTS} "
      f"({n_before} -> {panel_cum['district'].nunique()} districts in panel)")

# ── 4b. SYNTHETIC JAN-1 BASELINE SNAPSHOTS ──────────────────────────────────
years = sorted(panel_cum['year'].unique())
jan1 = pd.DataFrame([
    {'date': pd.Timestamp(y, 1, 1), 'year': y, 'district': d, 'cumulative': 0}
    for y in years for d in ACTIVE_DISTRICTS
])
panel_cum = pd.concat([panel_cum, jan1], ignore_index=True)
print(f"Injected {len(jan1)} synthetic Jan-1 zero snapshots "
      f"({len(years)} years x {len(ACTIVE_DISTRICTS)} districts)")

# ── 5. DERIVE INCIDENT CASES (first difference within year) ─────────────────
panel_cum = panel_cum.sort_values(['district','year','date']).reset_index(drop=True)
panel_cum['prev_cumulative'] = panel_cum.groupby(['district','year'])['cumulative'].shift(1)
panel_cum['incident'] = panel_cum['cumulative'] - panel_cum['prev_cumulative'].fillna(0)
panel_cum.to_csv(f'{OUT_DIR}/test_panel_file.csv',index=False)
neg_mask = panel_cum['incident'] < 0
if neg_mask.sum() > 0:
    print(f"\nWARNING: {neg_mask.sum()} negative incident values (data corrections) → clamped to 0")
    print(panel_cum[neg_mask][['date','district','cumulative','prev_cumulative','incident']].to_string())
    panel_cum.loc[neg_mask, 'incident'] = 0

panel_cum['incident'] = panel_cum['incident'].astype(int)

# Compute interval length (days between this and previous snapshot)
panel_cum['prev_date'] = panel_cum.groupby(['district','year'])['date'].shift(1)
panel_cum['interval_days'] = (panel_cum['date'] - panel_cum['prev_date']).dt.days.fillna(0).astype(int)
print(f"\nIncident derivation complete.")
print(f"Total incident cases across panel: {panel_cum['incident'].sum():,}")
print(f"Interval length stats (days): min={panel_cum['interval_days'].min()}, "
      f"max={panel_cum['interval_days'].max()}, "
      f"median={panel_cum['interval_days'].median():.0f}")

# ── 6. SAVE CUMULATIVE PANEL ─────────────────────────────────────────────────
out_cols = ['date','year','district','cumulative','incident','interval_days']
panel_cum[out_cols].to_csv(f'{OUT_DIR}/panel_cumulative.csv', index=False)
print(f"\nSaved: {OUT_DIR}/panel_cumulative.csv")

# ── 7. WEEKLY REGULARIZATION (ISO weeks) ────────────────────────────────────
panel_inc = panel_cum[panel_cum['interval_days'] > 0].copy()
print(panel_inc)

daily_records = []
for _, row in panel_inc.iterrows():
    if row['interval_days'] == 0:
        continue
    daily_rate = row['incident'] / row['interval_days']
    dates = pd.date_range(
        start=row['prev_date'] + pd.Timedelta(days=1),
        end=row['date'],
        freq='D'
    )
    for d in dates:
        daily_records.append({
            'date': d,
            'year': d.year,
            'district': row['district'],
            'daily_cases': daily_rate
        })

daily_panel = pd.DataFrame(daily_records)

# Aggregate to ISO weeks
daily_panel['iso_year'] = daily_panel['date'].dt.isocalendar().year.astype(int)
daily_panel['iso_week'] = daily_panel['date'].dt.isocalendar().week.astype(int)

weekly = (daily_panel
    .groupby(['district','iso_year','iso_week'])
    .agg(weekly_cases=('daily_cases','sum'))
    .reset_index()
)

# Add week start date for interpretability
weekly['week_start'] = weekly.apply(
    lambda r: datetime.fromisocalendar(int(r['iso_year']), int(r['iso_week']), 1), axis=1
)
weekly['week_start'] = pd.to_datetime(weekly['week_start'])
weekly = weekly.sort_values(['district','week_start']).reset_index(drop=True)
weekly['weekly_cases'] = weekly['weekly_cases'].round(2)

print(f"\nWeekly panel shape: {weekly.shape}")
print(f"No-case ISO weeks preserved as zeros: {(weekly['weekly_cases'] == 0).sum()}")
print(f"ISO weeks covered: {weekly['iso_week'].nunique()} unique weeks")
print(f"Week range: {weekly['week_start'].min().date()} → {weekly['week_start'].max().date()}")

weekly.to_csv(f'{OUT_DIR}/panel_weekly.csv', index=False)
print(f"Saved: {OUT_DIR}/panel_weekly.csv")

# ── 8. INGESTION REPORT ──────────────────────────────────────────────────────
report = []
report.append("=" * 70)
report.append("NEPAL DENGUE INGESTION REPORT (v_05)")
report.append("=" * 70)
report.append(f"Files processed     : {len(files)}")
report.append(f"Files skipped       : {len(skipped)}")
report.append(f"Schema - simple     : {schema_counts['simple']}")
report.append(f"Schema - wide       : {schema_counts['wide']}")
report.append(f"Date range          : {panel_cum['date'].min().date()} → {panel_cum['date'].max().date()}")
report.append(f"Districts (canonical) : {panel_cum['district'].nunique()} / 77 "
              f"(excluded by choice: {', '.join(EXCLUDED_DISTRICTS)})")
report.append(f"Total snapshots     : {len(panel_cum):,} (incl. {len(jan1)} synthetic Jan-1 baselines)")
report.append(f"Total incident cases: {panel_cum['incident'].sum():,}")
report.append(f"Negative incidents  : {neg_mask.sum()} (clamped to 0)")
report.append(f"Weekly panel rows   : {len(weekly):,}")
wk_mass = weekly.assign(yr=weekly['week_start'].dt.year).groupby('yr')['weekly_cases'].sum()
ann_mass = panel_cum.groupby('year')['incident'].sum()
report.append("")
report.append("MASS CONSERVATION (annual incident vs weekly sum by week_start year):")
for yr in ann_mass.index:
    shortfall = ann_mass[yr] - wk_mass.get(yr, 0.0)
    report.append(f"  {yr}: incident={ann_mass[yr]:,}  weekly={wk_mass.get(yr, 0.0):,.1f}  shortfall={shortfall:,.1f}")

report.append("")
report.append("INCIDENT DERIVATION: cumulative[t] - cumulative[t-1] within year.")
report.append("  Year boundary resets: a synthetic zero-cumulative snapshot is")
report.append("  injected at Jan 1 per district, so the first real snapshot keeps")
report.append("  incident = cumulative (YTD from Jan 1) and receives interval")
report.append("  Jan 1 -> snapshot date; January mass enters the weekly panel.")

report.append("")
report.append("WEEKLY REGULARIZATION: daily rate = incident / interval_days,")
report.append("  uniformly distributed across interval, then summed to ISO weeks.")
report.append("  Zero-incident intervals retained: no-case ISO weeks appear as")
report.append(f"  explicit zeros ({int((weekly['weekly_cases'] == 0).sum())} weeks), preserving seasonality.")

if skipped:
    report.append("\nSKIPPED FILES:")
    for f, reason in skipped:
        report.append(f"  {f}: {reason}")

report_text = "\n".join(report)
print("\n" + report_text)
with open(f'{OUT_DIR}/ingestion_report.txt','w', encoding='utf-8') as f:
    f.write(report_text)

print(f"\nSaved: {OUT_DIR}/ingestion_report.txt")
print("\n✓ Case count ingestion complete.")