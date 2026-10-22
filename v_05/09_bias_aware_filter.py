"""
09_bias_aware_filter.py
-----------------------

Output (v_05/outputs/):
    v05_bias_filter_results.csv   — per (cluster, model, condition) operational metrics
    v05_bias_filter_summary.csv   — mean RMSE per (cluster, model, condition) over draws/seeds
"""

import os
import numpy as np
import pandas as pd

from common import (ModelConfig, VanillaRNN, LSTMModel, fit_model, predict,
                           metrics, stage1_correct, load_bias_params, load_biased,
                           load_draw_params, load_cluster_clean, drop_excluded_districts,
                           per_cluster_years, TRAIN_YEARS, TEST_YEAR,
                           CONDITIONS, MODELS, align_to_truth, prediction_dates,
                           correct_full_then_split)

# Override cond/models for operational-only run
CONDITIONS = ['B1', 'B2', 'B3', 'combined']
MODELS = {'RNN': VanillaRNN, 'LSTM': LSTMModel}
TRAIN_FRAC = 0.70

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

V05_OUT = os.path.join(os.path.dirname(_SCRIPT_DIR), 'outputs')


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument('--draws', type=int, default=100, help='MC draws used')
    ap.add_argument('--seeds', type=int, default=2, help='seeds per draw')
    ap.add_argument('--clusters', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--conditions', type=str, nargs='+', default=CONDITIONS)
    args = ap.parse_args()

    np.random.seed(42)
    bp = load_bias_params()
    dp = load_draw_params()  # loaded but NOT used in operational mode

    # Load clean aggregate (for true 2025 test values — scoring only, NOT for correction)
    clean_agg, clusters_df = load_cluster_clean()
    weekly = pd.read_csv(os.path.join(V05_OUT, 'panel_weekly.csv'))
    weekly = drop_excluded_districts(weekly)
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    weekly = weekly.merge(clusters_df[['district', 'cluster']], on='district')

    # Load biased panels
    panels = {c: load_biased(c, n_draws=args.draws) for c in args.conditions}
    draws = sorted(panels[args.conditions[0]]['draw'].unique())

    rows = []
    for cluster in args.clusters:
        c24, c25 = per_cluster_years(clean_agg, cluster)
        nt = int(len(c24) * TRAIN_FRAC)
        clean_te = c25['weekly_cases'].values
        clean_te_dates = pd.to_datetime(c25['week_start'].values)

        for mname, ModelCls in MODELS.items():
            for cond in args.conditions:
                panel = panels[cond]
                for draw in draws:
                    d24 = panel[(panel['cluster'] == cluster) &
                                (panel['iso_year'].isin(TRAIN_YEARS)) &
                                (panel['draw'] == draw)].sort_values('week_start')
                    d25 = panel[(panel['cluster'] == cluster) &
                                (panel['iso_year'] == TEST_YEAR) &
                                (panel['draw'] == draw)].sort_values('week_start')
                    if len(d24) < 10 or len(d25) < 5:
                        continue

                    # ---- Full-series correction first, then leak-free split ----
                    # Correct the complete time-ordered biased series (train +
                    # test) so lag corrections keep cross-boundary history
                    # (C3 max lag 10wk vs T_IN 8), then partition. Chunk-wise
                    # correction would reset history at each split.
                    # The lag correction reads `lag` weeks into the future, so
                    # the last `lag` corrected weeks before each boundary are
                    # dropped (option B trim); the unfiltered arm is trimmed to
                    # the same calendar weeks for a fair comparison.
                    b_full = np.concatenate([d24['biased_cases'].values,
                                             d25['biased_cases'].values])
                    n24 = len(d24)
                    b_tr, b_rest = b_full[:nt], b_full[nt:]
                    b_va, b_te = b_rest[:n24 - nt], b_rest[n24 - nt:]
                    c_tr, c_va, c_te = correct_full_then_split(
                        b_full, nt, cluster, cond, bp, mode='operational',
                        draw_params=dp, draw=draw, ref=None,
                        n_val=n24 - nt)
                    b_tr, b_va = b_tr[:len(c_tr)], b_va[:len(c_va)]

                    b_te_dates = pd.to_datetime(d25['week_start'].values)

                    # Train on corrected training, evaluate on corrected test scored vs clean truth
                    for arm in ['operational']:
                        for seed in range(args.seeds):
                            m_uf, mu_uf, sd_uf = fit_model(b_tr, b_va, ModelCls, seed, cfg=ModelConfig())
                            m_f, mu_f, sd_f = fit_model(c_tr, c_va, ModelCls, seed, cfg=ModelConfig())
                            if m_uf is None or m_f is None:
                                continue

                            pred_uf = predict(m_uf, b_te, mu_uf, sd_uf, t_in=ModelConfig.t_in)
                            pred_f = predict(m_f, c_te, mu_f, sd_f, t_in=ModelConfig.t_in)
                            if len(pred_uf) < 2 or len(pred_f) < 2:
                                continue

                            # ---- Canonical date-aligned scoring ----
                            # Test set stays CLEAN truth; biased series is used
                            # for training/validation only. Both arms are joined
                            # to the same clean weeks on week_start, then
                            # restricted to their intersection so uf/filtered
                            # metrics always cover identical rows.
                            try:
                                d_uf, t_uf, p_uf = align_to_truth(
                                    clean_te_dates, clean_te,
                                    prediction_dates(b_te_dates, ModelConfig.t_in, len(pred_uf)),
                                    pred_uf)
                                d_f, t_f, p_f = align_to_truth(
                                    clean_te_dates, clean_te,
                                    prediction_dates(b_te_dates, ModelConfig.t_in, len(pred_f)),
                                    pred_f)
                            except ValueError:
                                continue
                            common = sorted(set(d_uf.tolist()) & set(d_f.tolist()))
                            if len(common) < 2:
                                continue
                            m_uf_idx = np.isin(d_uf, common)
                            m_f_idx = np.isin(d_f, common)
                            true = t_uf[m_uf_idx]
                            pu, pf = p_uf[m_uf_idx], p_f[m_f_idx]
                            n = len(common)

                            mu_ = metrics(true, pu)
                            mf_ = metrics(true, pf)
                            if mu_ is None or mf_ is None:
                                continue

                            rows.append({
                                'cluster': cluster, 'model': mname,
                                'condition': cond, 'arm': arm,
                                'draw': draw, 'seed': seed,
                                'eval_start': str(pd.to_datetime(common[0]).date()),
                                'eval_end': str(pd.to_datetime(common[-1]).date()),
                                'rmse_unfiltered': mu_['rmse'],
                                'rmse_filtered': mf_['rmse'],
                                'mae_unfiltered': mu_['mae'],
                                'mae_filtered': mf_['mae'],
                                'bias_unfiltered': mu_['bias'],
                                'bias_filtered': mf_['bias'],
                                'corr_unfiltered': mu_['corr'],
                                'corr_filtered': mf_['corr'],
                                'n_obs': n
                            })

    rdf = pd.DataFrame(rows)
    if len(rdf) > 0:
        rdf.to_csv(os.path.join(OUT_DIR, 'v05_bias_filter_results.csv'), index=False)
        print(f"Saved {len(rdf)} draw-level rows -> {OUT_DIR}/v05_bias_filter_results.csv")

        # Summary: mean improvement vs unfiltered, per cell
        rdf['improvement_pct'] = (rdf['rmse_unfiltered'] - rdf['rmse_filtered']) / rdf['rmse_unfiltered'] * 100
        g = (rdf.groupby(['cluster', 'model', 'condition'])
               .agg(rmse_unfiltered=('rmse_unfiltered', 'mean'),
                    rmse_filtered=('rmse_filtered', 'mean'),
                    improvement_pct=('improvement_pct', 'mean'),
                    sd_improvement=('improvement_pct', 'std'),
                    n=('improvement_pct', 'size'))
               .reset_index())
        g.to_csv(os.path.join(OUT_DIR, 'v05_bias_filter_summary.csv'), index=False)
        print("\n" + g.round(2).to_string(index=False))
        print(f"\nSaved summary -> {OUT_DIR}/v05_bias_filter_summary.csv")
        print("\nNOTE: improvement = (RMSE_uf - RMSE_f) / RMSE_uf * 100; "
              "operational arm uses prior means only (no clean data leakage).")
    else:
        print("No rows generated — check data availability and parameter ranges.")


if __name__ == '__main__':
    main()
