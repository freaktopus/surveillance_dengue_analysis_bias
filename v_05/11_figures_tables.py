"""
11_figures_tables.py
--------------------
Main manuscript figures for the v_05 pipeline (Nepal dengue forecasting
under surveillance bias).

Figures produced:
    Fig 1  Seasonal SEIR calibration + multi-lead rolling forecasts (2022-2025).
    Fig 2  RNN vs LSTM one-week-ahead forecasts under simulated surveillance
           bias, scored against the clean 2025 holdout (median +/- IQR).
    Fig 4  Bias-propagation tradeoff relative to the clean baseline across 24
           model-bias-tier combinations.

Style notes
-----------
* Colour palette is the colour-blind-aware Wong (Nature Methods) palette,
  mirrored across 07_model_diagnostics.py and v_01/make_paper_figures.py.
* Font is Helvetica via Nimbus Sans (URW metric-identical clone), registered
  explicitly so it embeds in PDF.
* Legend entries are explicit and descriptive so a reader can decode each
  series without the surrounding text.

Run:
    python 11_figures_tables.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs', '05b_model_training')
FIG_DIR = os.path.join(_SCRIPT_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Font setup: use Helvetica for all text. A real Helvetica TTF is not present,
# so Nimbus Sans (URW's metric-identical Helvetica clone) is registered and used
# as the fallback; the requested family stays "Helvetica" in the figure.
# ---------------------------------------------------------------------------
NIMBUS = [
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Italic.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-BoldItalic.otf",
]
for _p in NIMBUS:
    if os.path.exists(_p):
        fm.fontManager.addfont(_p)
# also try to add genuine Helvetica/pdfcorefonts metrics if matplotlib finds them
from matplotlib import font_manager as _fm
for _name in ("Helvetica", "Helvetica-Bold", "Helvetica-Oblique", "Helvetica-BoldOblique"):
    _p = f"/usr/share/matplotlib/mpl-data/fonts/pdfcorefonts/{_name}.afm"
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Helvetica",
    "mathtext.it": "Helvetica",
    "mathtext.bf": "Helvetica:bold",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 1.1,
    "axes.titlesize": 11,
    "axes.labelsize": 10.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
})

# ---------------------------------------------------------------------------
# Colour-blind-aware palette (Wong 2011, Nature Methods)
# ---------------------------------------------------------------------------
COND_COLORS = {
    "clean":    "#555555",   # universal grey baseline
    "B1":       "#0072B2",   # blue          - reporting-rate attenuation
    "B2":       "#D55E00",   # vermillion    - temporal lag
    "B3":       "#009E73",   # bluish-green  - ascertainment attenuation
    "combined": "#CC79A7",   # reddish-purple- combined B3 -> B1 -> B2
}
COND_LS = {"clean": (0, (5, 4)), "B1": "-", "B2": "-", "B3": "-", "combined": "-"}
COND_LABELS = {
    "clean":    "Clean baseline",
    "B1":       "B1 \u2192 reporting-rate attenuation",
    "B2":       "B2 \u2192 temporal lag",
    "B3":       "B3 \u2192 ascertainment attenuation",
    "combined": "Combined (B3 \u2192 B1 \u2192 B2)",
}
TIER_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
TIER_LABELS = {
    1: "C1 \u2014 Urban / high surveillance",
    2: "C2 \u2014 Peri-urban / mixed surveillance",
    3: "C3 \u2014 Rural / low surveillance",
}
TIER_SHORT = {1: "C1", 2: "C2", 3: "C3"}
TIER_DESC = {
    1: "Urban / high surveillance",
    2: "Peri-urban / mixed surveillance",
    3: "Rural / low surveillance",
}
LEAD_STYLE = {
    1: dict(color="#0072B2", ls="solid",  lw=0.9, marker="o", ms=4, mfc="none"),
    2: dict(color="#D55E00", ls="dashed", lw=0.8, marker="s", ms=4, mfc="none"),
    4: dict(color="#CC79A7", ls="dotted", lw=0.8, marker="^", ms=4, mfc="none"),
}
MODEL_MARKER = {"RNN": "o", "LSTM": "s"}

HOLD_START = pd.Timestamp("2025-01-01")
CAL_START = pd.Timestamp("2022-01-01")
HORIZON_K = 4
KASKI_NOTE = ("* Kaski district excluded from all panels: confirmed under-reporting "
              "with retrospective case-count correction \u22489 months after reporting.")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, name + "." + ext), bbox_inches="tight")
    print(f"  saved -> {FIG_DIR}/{name}.png/.pdf")


# ===========================================================================
# FIG 1 - Seasonal SEIR calibration + multi-lead rolling forecasts
# ===========================================================================
def fig1_seir_fit():
    print("[Fig 1] SEIR calibration + multi-lead rolling forecasts")
    fit = pd.read_csv(os.path.join(OUT_DIR, "seir_recalibrated_fitted.csv"))
    met = pd.read_csv(os.path.join(OUT_DIR, "seir_recalibrated_metrics.csv"))
    fit["week_start"] = pd.to_datetime(fit["week_start"])

    # calibration = 2022-2024 in-sample; rolling = 2025 out-of-sample
    cal = fit[(fit["split"] == "calibration") & (fit["week_start"] >= CAL_START)].copy()
    roll = fit[fit["split"] == "rolling_forecast"].copy()
    for c in [1, 2, 3]:
        m = (roll["cluster"] == c)
        order = np.arange(int(m.sum()))
        roll.loc[m, "lead"] = (order % HORIZON_K) + 1
    roll = roll[roll["week_start"] >= HOLD_START]

    fig = plt.figure(figsize=(7.4, 10.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 1.0],
                          hspace=0.42, wspace=0.16,
                          left=0.115, right=0.985, top=0.9, bottom=0.115)
    axcal = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axhol = [fig.add_subplot(gs[i, 1]) for i in range(3)]
    panel_cal = ["a", "c", "e"]
    panel_hol = ["b", "d", "f"]

    # ---- left column: calibration fit (full 2022-2024 scale) ----
    for i, c in enumerate([1, 2, 3]):
        ax = axcal[i]
        mc = met[met["cluster"] == c].iloc[0]
        cal_c = cal[cal["cluster"] == c].sort_values("week_start")
        ax.fill_between(cal_c["week_start"], cal_c["observed"],
                        color=TIER_COLORS[c], alpha=0.06, zorder=1)
        ax.plot(cal_c["week_start"], cal_c["observed"], "-", lw=0.8,
                color=TIER_COLORS[c], alpha=0.85, zorder=2,
                label="Observed cases (2022\u20132024)")
        ax.plot(cal_c["week_start"], cal_c["fitted"], "-", lw=1.0,
                color="#222222", alpha=0.7, zorder=3,
                label=f"SEIR calibration fit (R\u00b2 = {mc['cal_r2']:.2f})")
        ax.set_ylabel("Weekly cases", fontsize=9.5)
        ax.text(0.02, 0.97, panel_cal[i], transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="left", va="top")
        ax.set_title(f"{TIER_SHORT[c]} \u00b7 {TIER_DESC[c]}  "
                     f"\u2014  cal R\u00b2 = {mc['cal_r2']:.2f}",
                     fontsize=9.5, loc="left")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=8.5)
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        ax.tick_params(axis="x", which="minor", length=0)
        ax.grid(alpha=0.2, which="major", axis="y")

    # ---- right column: 2025 holdout with lead-1/2/4 forecasts ----
    for i, c in enumerate([1, 2, 3]):
        ax = axhol[i]
        mc = met[met["cluster"] == c].iloc[0]
        rol_c = roll[roll["cluster"] == c].sort_values("week_start")
        ax.fill_between(rol_c["week_start"], rol_c["observed"],
                        color=TIER_COLORS[c], alpha=0.07, zorder=1)
        ax.plot(rol_c["week_start"], rol_c["observed"], "-", lw=1.1,
                color="black", zorder=5, label="Observed cases (2025)")

        for lead, st in LEAD_STYLE.items():
            sub = rol_c[rol_c["lead"] == lead].sort_values("week_start")
            lbl = f"Lead-{lead} forecast"
            if lead == 1:
                lbl += f"  (R\u00b2 = {mc['holdout_r2_lead1']:.2f})"
            ax.plot(sub["week_start"], sub["fitted"], label=lbl, **st)

        ax.set_xlim(HOLD_START, pd.Timestamp("2025-12-31"))
        ytop = max(rol_c["observed"].max(), rol_c["fitted"].max()) * 1.15
        ax.set_ylim(bottom=0, top=ytop)
        ax.text(0.02, 0.97, panel_hol[i], transform=ax.transAxes, fontsize=12,
                fontweight="bold", ha="left", va="top")
        ax.set_title(f"2025 holdout \u2014 all-lead R\u00b2 = {mc['holdout_r2_all']:.2f}",
                     fontsize=9.5, loc="left")
        ax.tick_params(axis="y", labelsize=8.5)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax.tick_params(axis="x", labelsize=8.5)
        ax.grid(alpha=0.2, which="major", axis="y")
        if i == 2:
            ax.set_xlabel("2025", fontsize=9.5)

    # ---- single, accurate legend at the bottom ----
    handles = [
        Line2D([0], [0], color=TIER_COLORS[1], lw=0.8, alpha=0.85,
               label="Observed cases (2022\u20132024)"),
        Line2D([0], [0], color="#222222", lw=1.0, alpha=0.7,
               label="SEIR calibration fit"),
        Line2D([0], [0], color="black", lw=1.1, label="Observed cases (2025)"),
    ]
    for lead in (1, 2, 4):
        handles.append(Line2D([0], [0], **{**LEAD_STYLE[lead], "ms": 4},
                             label=f"Lead-{lead} forecast"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.015), handlelength=2.2)

    fig.suptitle("Seasonal SEIR calibration (2022\u20132024) and multi-lead rolling "
                 "forecasts across Nepal surveillance tiers (2025)",
                 fontsize=11.5, y=0.975)
    fig.text(0.5, 0.005, KASKI_NOTE, ha="center", va="bottom", fontsize=8,
             color="#333333")

    # axis labels for the shared columns handled per-row; add row head titles
    for i, c in enumerate([1, 2, 3]):
        axcal[i].set_ylabel(f"{TIER_SHORT[c]}\nWeekly cases", fontsize=9)

    save(fig, "fig1_seir_fit_leads")
    plt.close(fig)


# ===========================================================================
# FIG 2 - RNN vs LSTM one-week-ahead forecasts under simulated bias
# ===========================================================================
def fig2_rnn_lstm_biases():
    print("[Fig 2] RNN vs LSTM forecasts under simulated bias")
    p = pd.read_csv(os.path.join(OUT_DIR, "model_predictions.csv"))
    p["week_start"] = pd.to_datetime(p["week_start"])

    # score against the clean 2025 truth (observed is identical per cluster/week)
    obs = (p.groupby(["cluster", "week_start"])["observed"]
           .first().reset_index())

    fig, axes = plt.subplots(3, 2, figsize=(12.4, 10.8), sharex=True, sharey="row")
    n_weeks = int(obs["week_start"].nunique())

    for i, c in enumerate([1, 2, 3]):
        for j, model in enumerate(["RNN", "LSTM"]):
            ax = axes[i, j]
            ob = obs[obs["cluster"] == c].sort_values("week_start")
            ax.plot(ob["week_start"], ob["observed"], "-", color="black", lw=1.2,
                    zorder=20, label="Observed (clean 2025 truth)")

            for k, cond in enumerate(["clean", "B1", "B2", "B3", "combined"]):
                sub = p[(p["cluster"] == c) & (p["model"] == model)
                        & (p["condition"] == cond)].copy()
                if len(sub) == 0:
                    continue
                # median + IQR across the draw x seed ensemble
                grp = sub.groupby("week_start")["predicted"]
                agg = pd.DataFrame({
                    "med": grp.median(),
                    "lo": grp.quantile(0.25),
                    "hi": grp.quantile(0.75),
                }).sort_index()
                col = COND_COLORS[cond]
                ls = COND_LS[cond]
                z_base = 4 + 2 * k          # keep clean lowest, combined on top
                ax.fill_between(agg.index, agg["lo"], agg["hi"], color=col,
                                alpha=0.10, lw=0, zorder=z_base,
                                label=None)
                ax.plot(agg.index, agg["med"], ls=ls, color=col, lw=1.2,
                        zorder=z_base + 1, label=COND_LABELS[cond])

            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.2, which="major", axis="y")
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(3, 6, 9, 12)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.tick_params(axis="x", labelsize=8.5)
            if j == 0:
                ax.set_ylabel("Weekly cases", fontsize=9.5)
            if i == 0:
                ax.set_title(model, fontsize=12, fontweight="bold")
            if i == 2 and j == 0:
                ax.set_xlabel("")
            if i == 2 and j == 1:
                ax.set_xlabel("2025")

    # ---- legend ----
    handles, labels = axes[0, 0].get_legend_handles_labels()
    keeps = list(dict.fromkeys(labels))
    handle_map = dict(zip(labels, handles))
    order = ["Observed (clean 2025 truth)"] + [COND_LABELS[k] for k in COND_LABELS]
    new_handles = [handle_map[k] for k in order if k in handle_map]
    fig.legend(new_handles, order, loc="lower center", ncol=3, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01), handlelength=2.2)

    # ---- row (tier) labels ----
    for i, c in enumerate([1, 2, 3]):
        axes[i, 0].annotate(TIER_LABELS[c], xy=(0.008, 0.955),
                            xycoords="axes fraction", ha="left", va="top",
                            fontsize=9, color=TIER_COLORS[c],
                            zorder=15,
                            bbox=dict(boxstyle="round,pad=0.28", fc="white",
                                      ec="none", alpha=0.85))

    fig.suptitle("One-week-ahead forecasts (median \u00b1 IQR shade) across the "
                 f"{n_weeks}-week clean 2025 holdout \u2014 RNN (left) vs LSTM (right);"
                 "\nIQR over 100 bias draws \u00d7 3 seeds (clean: 10 seeds)",
                 fontsize=11.5, y=0.99)
    fig.tight_layout(rect=(0, 0.035, 1, 0.965))
    save(fig, "fig2_rnn_lstm_biases")
    plt.close(fig)


# ===========================================================================
# FIG 4 - Bias-propagation tradeoff relative to clean baseline
# ===========================================================================
def _compute_delta_corr(p):
    """Clean-baseline correlation per (cluster, model) from the clean condition."""
    clean = p[p["condition"] == "clean"]
    rows = []
    for (c, m), g in clean.groupby(["cluster", "model"]):
        corrs = []
        for seed in g["seed"].unique():
            sub = g[g["seed"] == seed]
            if sub["observed"].std() > 0 and sub["predicted"].std() > 0:
                corrs.append(np.corrcoef(sub["observed"], sub["predicted"])[0, 1])
        rows.append({"cluster": c, "model": m, "clean_corr": np.nanmean(corrs)})
    return pd.DataFrame(rows)


def _place_point_labels(ax, df, xkey, ykey, renderer, fontsize=5.8):
    """Place a small, thin label at EVERY point, minimising overlap."""
    placed = []
    label_col = "#333333"
    # group centroids -> seed each label outward from its own group centre, so the
    # four same-(tier, model) points push in different directions.
    centroids = df.groupby(["cluster", "model"])[[xkey, ykey]].mean().to_dict("index")
    group_counts = df.groupby(["cluster", "model"])[xkey].transform("count")
    group_rank = df.groupby(["cluster", "model"]).cumcount()
    for i, (idx, r) in enumerate(df.iterrows()):
        c, m = r["cluster"], r["model"]
        label = f"C{int(c)}{m[0]}"
        x, y = r[xkey], r[ykey]
        cd = centroids[(c, m)]
        vx, vy = x - cd[xkey], y - cd[ykey]
        norm = max(np.hypot(vx, vy), 1e-9)
        seed_ang = np.arctan2(vy / norm, vx / norm)
        # spread the (up to four) same-group points over distinct quadrant starts
        seed_ang += (group_rank[idx] / group_counts[idx]) * 2 * np.pi
        best = None
        # start far enough to clear the marker footprint, then spiral outward from
        # the group-outward direction so labels sit just above/around, not on, points
        for dist in np.arange(8.0, 55, 1.3):
            for d_ang in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                ang = seed_ang + d_ang
                ann = ax.annotate(label, xy=(x, y),
                                  xytext=(dist * np.cos(ang), dist * np.sin(ang)),
                                  textcoords="offset points", fontsize=fontsize,
                                  color=label_col, ha="center", va="center",
                                  zorder=10, fontweight="normal")
                bb = ann.get_window_extent(renderer=renderer)
                if not any(bb.overlaps(e) for e in placed):
                    best = (ann, bb)
                    break
                ann.remove()
            if best is not None:
                break
        if best is not None:
            placed.append(best[1])
        else:
            ang = seed_ang
            ann = ax.annotate(label, xy=(x, y),
                              xytext=(40 * np.cos(ang), 40 * np.sin(ang)),
                              textcoords="offset points", fontsize=fontsize,
                              color=label_col, ha="center", va="center",
                              zorder=10, fontweight="normal")
            placed.append(ann.get_window_extent(renderer=renderer))


def fig4_bias_tradeoff():
    print("[Fig 4] Bias-propagation tradeoff vs clean baseline")
    s = pd.read_csv(os.path.join(OUT_DIR, "bias_propagation_summary.csv"))
    p = pd.read_csv(os.path.join(OUT_DIR, "model_predictions.csv"))

    clean_corr = _compute_delta_corr(p)
    df = s.merge(clean_corr, on=["cluster", "model"])
    df["d_corr"] = df["mean_corr"] - df["clean_corr"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    cond_order = ["B1", "B2", "B3", "combined"]

    panels = [
        ("rmse_degradation_pct", "d_corr",
         "Relative RMSE change vs clean baseline (%)",
         "\u0394 Pearson correlation (bias condition \u2212 clean)",
         "(a)  \u0394corr vs relative RMSE change"),
        ("rmse_degradation_pct", "mean_forecast_bias",
         "Relative RMSE change vs clean baseline (%)",
         "Mean signed forecast error (cases/week)",
         "(b)  Level-shift: RMSE change vs forecast bias"),
    ]

    for ax, (xkey, ykey, xlab, ylab, title) in zip(axes, panels):
        for cond in cond_order:
            sub = df[df["condition"] == cond]
            for model in ["RNN", "LSTM"]:
                ss = sub[sub["model"] == model]
                ax.scatter(ss[xkey], ss[ykey], color=COND_COLORS[cond],
                           marker=MODEL_MARKER[model], s=62,
                           edgecolor="black", linewidth=0.4, zorder=5,
                           label=None)
        ax.axvline(0, color="#888888", lw=1.0, ls=(0, (4, 4)), zorder=1)
        if ykey == "d_corr":
            ax.axhline(0, color="#888888", lw=1.0, ls=(0, (4, 4)), zorder=1)
        else:
            ax.axhline(0, color="#aaaaaa", lw=0.8, ls=":", zorder=1)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(alpha=0.25, which="both", zorder=0)

    # ---- legend: condition colours + model markers ----
    cond_handles = [Patch(facecolor=COND_COLORS[c], edgecolor="none",
                          label=COND_LABELS[c]) for c in cond_order]
    model_handles = [Line2D([0], [0], color="#555555", marker=m, ls="", ms=8,
                            label=md) for md, m in MODEL_MARKER.items()]
    fig.legend(handles=cond_handles + model_handles, loc="lower center",
               ncol=6, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, -0.085),
               handlelength=1.2)
    fig.suptitle("Bias-propagation tradeoff relative to clean baseline \u2014 "
                 "24 model-bias-tier combinations (2025 holdout)",
                 fontsize=12, y=0.985)

    # place labels AFTER final layout so the overlap check uses final geometry
    fig.tight_layout(rect=(0, 0.075, 1, 0.965))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, (xkey, ykey, _, _, _) in zip(axes, panels):
        _place_point_labels(ax, df, xkey, ykey, renderer)
    save(fig, "fig4_bias_tradeoff")
    plt.close(fig)


if __name__ == "__main__":
    print("Building v_05 manuscript figures (Wong colour-blind palette)\n")
    fig1_seir_fit()
    fig2_rnn_lstm_biases()
    fig4_bias_tradeoff()
    print("\nDone.")
