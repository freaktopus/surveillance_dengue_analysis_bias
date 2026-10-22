"""
10_adaptive_lambda_shrinkage.py
--------------------------------
v_05 Adaptive Lambda Shrinkage

The rule: lambda = clamp(1 - kappa * CV(correction), 0, 1)
  where CV is the coefficient of variation of the multiplicative correction
  implied by the prior distribution.

Operational only: no oracle grid-search on clean tuning split. The gap
between rule-lambda and oracle-lambda results measures clean-label leakage.

Output (v_05/outputs/):
    v05_lambda_rule.csv     — a-priori lambda per (cluster, condition) + CVs
    v05_lambda_results.csv  — draw-level test metrics under rule-lambda
"""

import os
import numpy as np
import pandas as pd

from common import (ModelConfig, VanillaRNN, LSTMModel, fit_model, predict,
                   prior_cv_scale, lambda_rule, stage1_correct, metrics, load_bias_params,
                   load_biased, load_cluster_clean, drop_excluded_districts,
                   per_cluster_years, TRAIN_YEARS, TEST_YEAR, CONDITIONS, MODELS,
                   align_to_truth, prediction_dates, correct_full_then_split)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(_SCRIPT_DIR, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

V05_OUT = os.path.join(os.path.dirname(_SCRIPT_DIR), 'outputs')


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument('--draws', type=int, default=100, help='MC draws used')
    ap.add_argument('--seeds', type=int, default=2, help='seeds per draw')
    ap.add_argument('--clusters', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--conditions', type=str, nargs='+', default=['B1', 'B2', 'B3', 'combined'])
    args = ap.parse_args()

    np.random.seed(42)
    bp = load_bias_params()

    # Load clean aggregate (for true 2025 test values — scoring only)
    clean_agg, clusters_df = load_cluster_clean()
    weekly = pd.read_csv(os.path.join(V05_OUT, 'panel_weekly.csv'))
    weekly = drop_excluded_districts(weekly)
    weekly['week_start'] = pd.to_datetime(weekly['week_start'])
    weekly = weekly.merge(clusters_df[['district', 'cluster']], on='district')

    # Load biased panels
    panels = {c: load_biased(c, n_draws=args.draws) for c in args.conditions}
    draws = sorted(panels[args.conditions[0]]['draw'].unique())

    # 1. Write a-priori lambda rule table (from priors only, no clean data)
    rule_rows = []
    for cluster in args.clusters:
        for cond in args.conditions:
            lam = lambda_rule(cluster, cond, bp)
            cv, _ = prior_cv_scale(cluster, cond, bp)
            rule_rows.append({'cluster': cluster, 'condition': cond,
                              'cv_prior': cv, 'lambda_rule': lam})
    rule_df = pd.DataFrame(rule_rows)
    rule_df.to_csv(os.path.join(OUT_DIR, 'v05_lambda_rule.csv'), index=False)
    print("A-priori lambda rule (from priors only):")
    print(rule_df.round(3).to_string(index=False))

    # 2. Draw-level test metrics under rule-lambda (operational only)
    rows = []
    for cluster in args.clusters:
        for cond in args.conditions:
            lam = lambda_rule(cluster, cond, bp)
            c24, c25 = per_cluster_years(clean_agg, cluster)
            nt = int(len(c24) * 0.70)
            # Clean 2025 is SCORING-ONLY truth (never for training/validation).
            # Validation must match the training distribution (biased->biased,
            # corrected->corrected); validating a biased-trained model on clean
            # data leaks the future clean distribution into model selection.
            cl_te = c25['weekly_cases'].values  # 2025 holdout truth
            cl_te_dates = pd.to_datetime(c25['week_start'].values)

            for mname, ModelCls in MODELS.items():
                panel = panels[cond]
                for draw in draws:
                    d24 = panel[(panel['cluster'] == cluster) &
                                (panel['iso_year'].isin(TRAIN_YEARS)) &
                                (panel['draw'] == draw)].sort_values('week_start')
                    d25 = panel[(panel['cluster'] == cluster) &
                                (panel['iso_year'] == TEST_YEAR) &
                                (panel['draw'] == draw)].sort_values('week_start')
                    if len(d24) < nt + 5 or len(d25) < 5:
                        continue

                    # Full-series correction first, then leak-free split (cross-
                    # boundary lag history preserved; last `lag` corrected weeks
                    # before each boundary dropped — see 09 for rationale).
                    b_full = np.concatenate([d24['biased_cases'].values,
                                             d25['biased_cases'].values])
                    n24 = len(d24)
                    b_tr, b_rest = b_full[:nt], b_full[nt:]
                    b_va, b_te = b_rest[:n24 - nt], b_rest[n24 - nt:]
                    c_tr, c_va, c_te = correct_full_then_split(
                        b_full, nt, cluster, cond, bp, mode='operational',
                        n_val=n24 - nt)
                    b_tr, b_va = b_tr[:len(c_tr)], b_va[:len(c_va)]
                    b_te_dates = pd.to_datetime(d25['week_start'].values)

                    for seed in range(args.seeds):
                        m_uf, mu_uf, sd_uf = fit_model(b_tr, b_va, ModelCls, seed, cfg=ModelConfig())
                        m_f, mu_f, sd_f = fit_model(c_tr, c_va, ModelCls, seed, cfg=ModelConfig())
                        if m_uf is None or m_f is None:
                            continue

                        pu_te = predict(m_uf, b_te, mu_uf, sd_uf, t_in=ModelConfig.t_in)
                        pf_te = predict(m_f, c_te, mu_f, sd_f, t_in=ModelConfig.t_in)
                        if len(pu_te) < 2 or len(pf_te) < 2:
                            continue

                        # Canonical date-aligned scoring on the shared clean
                        # weeks; ensemble built on the intersection so every
                        # arm covers identical rows.
                        try:
                            d_u, t_u, p_u = align_to_truth(
                                cl_te_dates, cl_te,
                                prediction_dates(b_te_dates, ModelConfig.t_in, len(pu_te)),
                                pu_te)
                            d_f, _, p_f = align_to_truth(
                                cl_te_dates, cl_te,
                                prediction_dates(b_te_dates, ModelConfig.t_in, len(pf_te)),
                                pf_te)
                        except ValueError:
                            continue
                        common = sorted(set(d_u.tolist()) & set(d_f.tolist()))
                        if len(common) < 2:
                            continue

                        # Ensure same length for ensemble
                        n_common = len(common)
                        t = t_u[np.isin(d_u, common)]
                        pu = p_u[np.isin(d_u, common)]
                        pf = p_f[np.isin(d_f, common)]
                        ens_rule = lam * pf + (1 - lam) * pu
                        # Unfiltered (no correction) for comparison
                        mu_ = metrics(t, pu)

                        if mu_ is None:
                            continue

                        rows.append({
                            'cluster': cluster, 'model': mname,
                            'condition': cond, 'draw': draw, 'seed': seed,
                            'lambda_rule': lam,
                            'eval_start': str(pd.to_datetime(common[0]).date()),
                            'eval_end': str(pd.to_datetime(common[-1]).date()),
                            'n_obs': n_common,
                            'rmse_unfiltered': mu_['rmse'],
                            'rmse_rule': metrics(t, ens_rule)['rmse'] if len(ens_rule) >= 2 else np.nan,
                            'mae_rule': np.mean(np.abs(t - ens_rule)) if len(ens_rule) >= 2 else np.nan,
                        })

    rdf = pd.DataFrame(rows)
    if len(rdf) > 0:
        rdf.to_csv(os.path.join(OUT_DIR, 'v05_lambda_results.csv'), index=False)
        print(f"\nSaved {len(rdf)} rows -> {OUT_DIR}/v05_lambda_results.csv")

        # Summary: mean improvement from rule vs unfiltered
        rdf['impr_rule'] = (rdf['rmse_unfiltered'] - rdf['rmse_rule']) / rdf['rmse_unfiltered'] * 100
        g = (rdf.groupby(['cluster', 'model', 'condition'])
               .agg(lam_rule=('lambda_rule', 'first'),
                    rmse_unfiltered=('rmse_unfiltered', 'mean'),
                    rmse_rule=('rmse_rule', 'mean'),
                    impr_rule=('impr_rule', 'mean'),
                    n=('impr_rule', 'size'))
               .reset_index())
        print("\n" + g.round(3).to_string(index=False))
        print(f"\nSaved summary -> {OUT_DIR}/v05_lambda_results_summary_improvement.csv (via groupby)")
        # Save explicit summary
        g.to_csv(os.path.join(OUT_DIR, 'v05_lambda_rule_summary.csv'), index=False)
    else:
        print("No rows generated — check data availability.")


if __name__ == '__main__':
    main()
