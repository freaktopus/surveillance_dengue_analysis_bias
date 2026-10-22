

import os
import numpy as np
import pandas as pd

from common import (ModelConfig, VanillaRNN, LSTMModel, fit_model, predict,
                    metrics, stage1_correct, load_bias_params, load_biased,
                    load_draw_params, load_cluster_clean, drop_excluded_districts,
                    per_cluster_years, nb_dispersion_mom, nb_nll,
                    TRAIN_YEARS, TEST_YEAR, CONDITIONS, align_to_truth,
                    prediction_dates, correct_full_then_split)

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
    ap.add_argument('--alpha-mode', choices=['mom', 'profile'], default='mom',
                    help="dispersion: 'mom' (clean-train MoM per cluster) or "
                         "'profile' (per-series MLE on the scored series)")
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

    # Leakage-free dispersion: per-cluster MoM alpha from the clean train window
    alpha_by_cluster = {}
    if args.alpha_mode == 'mom':
        for cluster in args.clusters:
            c24, _ = per_cluster_years(clean_agg, cluster)
            alpha_by_cluster[cluster] = nb_dispersion_mom(c24['weekly_cases'].values)
            print(f"cluster {cluster}: MoM alpha = {alpha_by_cluster[cluster]:.4f}")

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
                    # Same rationale as 09: lag corrections need unbroken
                    # history across the train/val/test boundaries, with the
                    # last `lag` corrected weeks before each boundary dropped.
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

                    for seed in range(args.seeds):
                        m_uf, mu_uf, sd_uf = fit_model(b_tr, b_va, ModelCls, seed, cfg=ModelConfig())
                        m_f, mu_f, sd_f = fit_model(c_tr, c_va, ModelCls, seed, cfg=ModelConfig())
                        if m_uf is None or m_f is None:
                            continue

                        pred_uf = predict(m_uf, b_te, mu_uf, sd_uf, t_in=ModelConfig.t_in)
                        pred_f = predict(m_f, c_te, mu_f, sd_f, t_in=ModelConfig.t_in)
                        if len(pred_uf) < 2 or len(pred_f) < 2:
                            continue

                        # ---- Canonical date-aligned scoring (clean test truth) ----
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

                        alpha = alpha_by_cluster.get(cluster)
                        nu_ = nb_nll(true, pu, alpha=alpha)
                        nf_ = nb_nll(true, pf, alpha=alpha)
                        if nu_ is None or nf_ is None:
                            continue

                        rows.append({
                            'cluster': cluster, 'model': mname,
                            'condition': cond, 'draw': draw, 'seed': seed,
                            'eval_start': str(pd.to_datetime(common[0]).date()),
                            'eval_end': str(pd.to_datetime(common[-1]).date()),
                            'nll_unfiltered': nu_['nb_nll'],
                            'nll_filtered': nf_['nb_nll'],
                            'alpha_unfiltered': nu_['alpha'],
                            'alpha_filtered': nf_['alpha'],
                            'n_obs': n,
                        })

    rdf = pd.DataFrame(rows)
    if len(rdf) > 0:
        rdf.to_csv(os.path.join(OUT_DIR, 'v05_bias_filter_nll_results.csv'), index=False)
        print(f"Saved {len(rdf)} draw-level rows -> {OUT_DIR}/v05_bias_filter_nll_results.csv")

        rdf['improvement_pct'] = (rdf['nll_unfiltered'] - rdf['nll_filtered']) / \
                                 rdf['nll_unfiltered'].abs() * 100
        g = (rdf.groupby(['cluster', 'model', 'condition'])
                .agg(nll_unfiltered=('nll_unfiltered', 'mean'),
                     nll_filtered=('nll_filtered', 'mean'),
                     improvement_pct=('improvement_pct', 'mean'),
                     sd_improvement=('improvement_pct', 'std'),
                     n=('improvement_pct', 'size'))
                .reset_index())
        g.to_csv(os.path.join(OUT_DIR, 'v05_bias_filter_nll_summary.csv'), index=False)
        print("\n" + g.round(3).to_string(index=False))
        print(f"\nSaved summary -> {OUT_DIR}/v05_bias_filter_nll_summary.csv")
        print("\nNOTE: NB-NLL with var = mu + alpha*mu^2; mu = model prediction, "
              "y = clean 2025 weekly counts. improvement = (NLL_uf - NLL_f)/|NLL_uf| * 100; "
              "positive = filtering improves (lower) NLL. "
              "Operational arm uses prior means only (no clean data leakage).")
    else:
        print("No rows generated — check data availability and parameter ranges.")


if __name__ == '__main__':
    main()
