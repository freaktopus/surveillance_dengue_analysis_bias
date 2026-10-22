"""
panel_generator_kaski_v_01.py
----------------------------
Kaski-only weekly panel generator (2022-01-01 to 2024-10-21).

Working logic mirrored from v_05/03_case_count_ingestion.py:
  1. Read case snapshot CSVs from repo/data/ (filenames are YYYY-MM-DD.csv).
     Two schemas handled identically to the original:
       - simple: [District, TOTAL]            (no PROVINCE column)
       - wide:   [SN, PROVINCE, DISTRICT, JAN*..DEC, TOTAL] (has PROVINCE column)
     In both cases the TOTAL column is the year-to-date cumulative count.
  2. Keep ONLY district == 'KASKI' (original kept 76 districts, excluding KASKI;
     this script inverts that filter) and restrict snapshot dates to
     [2022-01-01, 2024-10-21].
  3. Inject synthetic Jan-1 zero-cumulative baselines for KASKI for each year
     in range (2022, 2023, 2024), exactly as the original does per district/year.
  4. Derive incident cases as first difference of cumulative within
     (district, year): incident[t] = cumulative[t] - cumulative[t-1],
     clamp negatives (data corrections) to 0, compute interval_days.
  5. Weekly regularization (identical to original): daily_rate =
     incident / interval_days uniformly spread over (prev_date+1 .. date),
     then summed to ISO weeks (iso_year, iso_week, week_start, weekly_cases).

Output: v_05/outputs/panel_weekly_kaski_v_01.csv
"""

import glob
import os
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_DIR = os.path.dirname(_SCRIPT_DIR)
DATA_DIR = os.path.join(_REPO_DIR, "data")
OUT_DIR = os.path.join(_SCRIPT_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Window + target (only differences vs 03_case_count_ingestion.py) ──
TARGET_DISTRICT = "KASKI"
START_DATE = pd.Timestamp("2022-01-01")
END_DATE = pd.Timestamp("2024-10-21")
OUT_WEEKLY = os.path.join(OUT_DIR, "panel_weekly_kaski_v_01.csv")
OUT_CUMULATIVE = os.path.join(OUT_DIR, "panel_cumulative_kaski_v_01.csv")


# ── Schema normalizers (copied from 03_case_count_ingestion.py) ──
def normalize_district(name):
    if pd.isna(name):
        return None
    n = str(name).strip().upper()
    aliases = {
        "ALL": "__DROP__",
        "TOTAL": "__DROP__",
        "GRAND TOTAL": "__DROP__",
    }
    return aliases.get(n, n)


def parse_simple(df, date):
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={"DISTRICT": "district", "TOTAL": "cumulative"})
    df["district"] = df["district"].apply(normalize_district)
    df = df[~df["district"].isin(["__DROP__", None])]
    df["date"] = date
    df["year"] = date.year
    return df[["date", "year", "district", "cumulative"]]


def parse_wide(df, date):
    df.columns = [c.strip().upper() for c in df.columns]
    df = df.rename(columns={"DISTRICT": "district", "TOTAL": "cumulative"})
    df["district"] = df["district"].apply(normalize_district)
    df = df[~df["district"].isin(["__DROP__", None])]
    df["date"] = date
    df["year"] = date.year
    return df[["date", "year", "district", "cumulative"]]


# ── Load all files in window ──
files = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
print(f"Found {len(files)} files in data/")

records = []
skipped = []
schema_counts = {"simple": 0, "wide": 0}

for fpath in files:
    fname = os.path.basename(fpath)
    try:
        date = pd.to_datetime(fname.replace(".csv", ""))
    except Exception:
        skipped.append((fname, "unparseable date"))
        continue

    # Window filter: only 2022-01-01 .. 2024-10-21
    if date < START_DATE or date > END_DATE:
        continue

    try:
        raw = pd.read_csv(fpath)
    except Exception as e:
        skipped.append((fname, f"read error: {e}"))
        continue

    cols_upper = [c.strip().upper() for c in raw.columns]
    if "PROVINCE" in cols_upper:
        parsed = parse_wide(raw, date)
        schema_counts["wide"] += 1
    else:
        parsed = parse_simple(raw, date)
        schema_counts["simple"] += 1

    # Kaski-only filter (applied early per file, same result as post-filter)
    parsed = parsed[parsed["district"] == TARGET_DISTRICT]
    if len(parsed) == 0:
        skipped.append((fname, "no KASKI row"))
        continue
    records.append(parsed)

if not records:
    raise SystemExit("No KASKI records found in window — check data/ filenames.")

panel_cum = pd.concat(records, ignore_index=True)
panel_cum["cumulative"] = (
    pd.to_numeric(panel_cum["cumulative"], errors="coerce").fillna(0).astype(int)
)
# De-duplicate in case of duplicate snapshot dates, keep last
panel_cum = panel_cum.sort_values("date").drop_duplicates(
    subset=["district", "date"], keep="last"
)

print(f"\nSchema counts (in-window): {schema_counts}")
print(f"KASKI snapshots (raw): {panel_cum.shape}")
print(
    f"Date range: {panel_cum['date'].min().date()} → {panel_cum['date'].max().date()}"
)

# ── Synthetic Jan-1 baseline snapshots (same logic as original) ──
years = sorted(panel_cum["year"].unique())
jan1 = pd.DataFrame(
    [
        {
            "date": pd.Timestamp(y, 1, 1),
            "year": y,
            "district": TARGET_DISTRICT,
            "cumulative": 0,
        }
        for y in years
    ]
)
panel_cum = pd.concat([panel_cum, jan1], ignore_index=True)
print(f"Injected {len(jan1)} synthetic Jan-1 zero snapshots for KASKI (years={years})")

# ── Derive incident cases (first difference within year) ──
panel_cum = panel_cum.sort_values(["district", "year", "date"]).reset_index(drop=True)
panel_cum["prev_cumulative"] = panel_cum.groupby(["district", "year"])[
    "cumulative"
].shift(1)
panel_cum["incident"] = (
    panel_cum["cumulative"] - panel_cum["prev_cumulative"].fillna(0)
)

neg_mask = panel_cum["incident"] < 0
if neg_mask.sum() > 0:
    print(
        f"\nWARNING: {neg_mask.sum()} negative incident values "
        f"(data corrections) → clamped to 0"
    )
    print(
        panel_cum[neg_mask][
            ["date", "district", "cumulative", "prev_cumulative", "incident"]
        ].to_string()
    )
    panel_cum.loc[neg_mask, "incident"] = 0

panel_cum["incident"] = panel_cum["incident"].astype(int)

panel_cum["prev_date"] = panel_cum.groupby(["district", "year"])["date"].shift(1)
panel_cum["interval_days"] = (
    (panel_cum["date"] - panel_cum["prev_date"]).dt.days.fillna(0).astype(int)
)
print(f"\nTotal KASKI incident cases in window: {panel_cum['incident'].sum():,}")

out_cols = ["date", "year", "district", "cumulative", "incident", "interval_days"]
panel_cum[out_cols].to_csv(OUT_CUMULATIVE, index=False)
print(f"Saved: {OUT_CUMULATIVE}")

# ── Weekly regularization (ISO weeks; identical to original) ──
panel_inc = panel_cum[panel_cum["interval_days"] > 0].copy()

daily_records = []
for _, row in panel_inc.iterrows():
    if row["interval_days"] == 0:
        continue
    daily_rate = row["incident"] / row["interval_days"]
    dates = pd.date_range(
        start=row["prev_date"] + pd.Timedelta(days=1),
        end=row["date"],
        freq="D",
    )
    for d in dates:
        daily_records.append(
            {
                "date": d,
                "year": d.year,
                "district": row["district"],
                "daily_cases": daily_rate,
            }
        )

daily_panel = pd.DataFrame(daily_records)
daily_panel["iso_year"] = daily_panel["date"].dt.isocalendar().year.astype(int)
daily_panel["iso_week"] = daily_panel["date"].dt.isocalendar().week.astype(int)

weekly = (
    daily_panel.groupby(["district", "iso_year", "iso_week"])
    .agg(weekly_cases=("daily_cases", "sum"))
    .reset_index()
)

weekly["week_start"] = weekly.apply(
    lambda r: datetime.fromisocalendar(
        int(r["iso_year"]), int(r["iso_week"]), 1
    ),
    axis=1,
)
weekly["week_start"] = pd.to_datetime(weekly["week_start"])
weekly = weekly.sort_values(["district", "week_start"]).reset_index(drop=True)
weekly["weekly_cases"] = weekly["weekly_cases"].round(2)

# Safety: daily mass only spans the window, but drop any week starting after END_DATE
weekly = weekly[weekly["week_start"] <= END_DATE].reset_index(drop=True)

print(f"\nWeekly panel shape: {weekly.shape}")
print(f"Week range: {weekly['week_start'].min().date()} → "
      f"{weekly['week_start'].max().date()}")
print(f"Sum weekly_cases: {weekly['weekly_cases'].sum():,.1f} "
      f"(incident total: {panel_cum['incident'].sum():,})")

weekly.to_csv(OUT_WEEKLY, index=False)
print(f"Saved: {OUT_WEEKLY}")
print("\nKaski weekly panel generation complete.")
