"""
05_model_selection.py
---------------------
Hyperparameter selection for RNN/LSTM dengue models using a
WALK-FORWARD protocol on 2022-2024 ONLY.

Modes:
  --mode ablation (default): efficient suite to test hidden vs others
    1) hidden sweep at baseline epochs/LR/batch (original)
    2) training sweep at hidden=4 (epochs x LR x batch)
    3) interaction grid hidden x LR and hidden x epochs (small factorial)
  --mode full: exhaustive cartesian product hidden*epochs*LR*batch
  --mode hidden_only: original behaviour for backwards compat

Outputs (v_05/outputs/05b_model_training):
    model_selection_walkforward.csv  -- full factorial rows (all configs)
    model_selection_summary.csv      -- marginal + best per hidden / per config
    figures/fig_model_selection.png
    figures/fig_interaction_hidden_lr.png
    figures/fig_interaction_hidden_epochs.png
    figures/fig_underfit_diagnostic.png
"""

import argparse
import itertools
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
V05_OUT = os.path.join(_SCRIPT_DIR, 'outputs')
MODEL_OUT = os.path.join(V05_OUT, '05b_model_training')
FIG_DIR = os.path.join(MODEL_OUT, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_OUT, exist_ok=True)

# ---- CLI --------------------------------------------------------------------
argp = argparse.ArgumentParser(description='Walk-forward hyperparameter selection (2022-2024 only)')
argp.add_argument('--seeds', type=int, default=3, help='seeds per (fold,cluster,config)')
argp.add_argument('--hidden', type=int, nargs='+', default=[4, 8, 16, 32, 64])
argp.add_argument('--epochs', type=int, nargs='+', default=[50, 100, 150, 200])
argp.add_argument('--lrs', type=float, nargs='+', default=[0.0005, 0.001, 0.005])
argp.add_argument('--batches', type=int, nargs='+', default=[8, 16, 32])
argp.add_argument('--mode', choices=['ablation', 'full', 'hidden_only'], default='ablation',
                  help='ablation=efficient hidden vs training sweep, full=exhaustive, hidden_only=original')
argp.add_argument('--max_combos', type=int, default=None, help='cap combos for full mode (random sample)')
args = argp.parse_args()
N_SEEDS = args.seeds
HIDDEN_GRID = args.hidden
EPOCHS_GRID = args.epochs
LR_GRID = args.lrs
BATCH_GRID = args.batches
MODE = args.mode

T_IN = 8
TRAIN_YEARS = [2022, 2023, 2024]
FOLDS = [2023, 2024]
ES_FRAC = 0.25
torch.set_num_threads(4)
np.random.seed(42)

# baseline (original) for hidden_only and as reference in ablation
BASE_EPOCHS, BASE_LR, BASE_BATCH = 150, 0.001, 16

# ---- MODELS -----------------------------------------------------------------
def make_model(kind, hidden):
    if kind == 'RNN':
        return nn.RNN(1, hidden, batch_first=True), nn.Linear(hidden, 1)
    else:
        return nn.LSTM(1, hidden, batch_first=True), nn.Linear(hidden, 1)

EXCLUDED_DISTRICTS = ['KASKI']

# ---- DATA -------------------------------------------------------------------
weekly = pd.read_csv(f'{V05_OUT}/panel_weekly.csv')
clusters_df = pd.read_csv(f'{V05_OUT}/district_clusters.csv')
weekly = weekly[~weekly['district'].isin(EXCLUDED_DISTRICTS)].copy()
clusters_df = clusters_df[~clusters_df['district'].isin(EXCLUDED_DISTRICTS)].copy()
weekly['week_start'] = pd.to_datetime(weekly['week_start'])
weekly = weekly.merge(clusters_df[['district', 'cluster']], on='district')

clean_agg = (weekly.groupby(['cluster', 'iso_year', 'iso_week', 'week_start'])
             ['weekly_cases'].sum().reset_index()
             .sort_values(['cluster', 'week_start']).reset_index(drop=True))

def make_sequences(series, t_in=T_IN):
    X, y = [], []
    for i in range(len(series) - t_in):
        X.append(series[i:i + t_in])
        y.append(series[i + t_in])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ---- TRAIN / PREDICT --------------------------------------------------------
def train_model(train, es_val, hidden, kind, seed, epochs, lr, batch):
    """Train with early stopping on es_val. Returns (model, mu, std, best_epoch, train_rmse)."""
    torch.manual_seed(seed); np.random.seed(seed)
    mu, std = train.mean() + 1e-8, train.std() + 1e-8
    norm = lambda x: (x - mu) / std
    Xtr, ytr = make_sequences(norm(train))
    Xva, yva = make_sequences(norm(es_val))
    if len(Xtr) < 4 or len(Xva) < 1:
        return None, None, None, 0, np.nan

    rnn, lin = make_model(kind, hidden)

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = rnn
            self.fc = lin
        def forward(self, x):
            out, _ = self.rnn(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    model = _Net()
    loader = DataLoader(TensorDataset(torch.tensor(Xtr).unsqueeze(-1), torch.tensor(ytr)),
                        batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best_vl, best_st, pat, best_ep = np.inf, None, 0, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = loss_fn(model(torch.tensor(Xva).unsqueeze(-1)), torch.tensor(yva)).item()
        if vl < best_vl - 1e-9:
            best_vl = vl
            best_st = {k: v.clone() for k, v in model.state_dict().items()}
            pat = 0
            best_ep = ep + 1
        else:
            pat += 1
            if pat >= 20:
                break
    if best_st is None:
        return None, None, None, best_ep, np.nan
    model.load_state_dict(best_st)
    # train RMSE diagnostic (in original scale) for underfit check
    model.eval()
    with torch.no_grad():
        tr_pred = model(torch.tensor(Xtr).unsqueeze(-1)).numpy() * std + mu
        tr_true = ytr * std + mu  # ytr already normed? reconstruct: ytr = (true-mu)/std
        # ytr was normed, so denorm
        # Actually ytr above is normed; use denormed arrays
    # recompute denormed correctly
    tr_true_denorm = ytr * std + mu
    train_rmse = np.sqrt(np.mean((tr_true_denorm - tr_pred)**2)) if len(tr_pred) > 0 else np.nan
    return model, mu, std, best_ep, train_rmse


def predict(model, series, mu, std):
    if model is None:
        return np.array([])
    model.eval()
    z = (series - mu) / std
    X, _ = make_sequences(z)
    if len(X) == 0:
        return np.array([])
    with torch.no_grad():
        pred = model(torch.tensor(X).unsqueeze(-1)).numpy()
    return pred * std + mu

# ---- BUILD COMBINATORIAL GRID ------------------------------------------------
def build_grid():
    if MODE == 'hidden_only':
        combos = [(h, BASE_EPOCHS, BASE_LR, BASE_BATCH) for h in HIDDEN_GRID]
    elif MODE == 'full':
        combos = list(itertools.product(HIDDEN_GRID, EPOCHS_GRID, LR_GRID, BATCH_GRID))
        if args.max_combos is not None and len(combos) > args.max_combos:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(combos), args.max_combos, replace=False)
            combos = [combos[i] for i in sorted(idx)]
    else:  # ablation - efficient factorial to test underfit
        # 1) hidden sweep at baseline
        s1 = [(h, BASE_EPOCHS, BASE_LR, BASE_BATCH) for h in HIDDEN_GRID]
        # 2) training sweep at hidden=4 (tests if h=4 underfit due to budget)
        s2 = [(4, e, lr, b) for e in EPOCHS_GRID for lr in LR_GRID for b in BATCH_GRID]
        # 3) interaction: hidden x LR at baseline epochs/batch
        s3 = [(h, BASE_EPOCHS, lr, BASE_BATCH) for h in [4, 16, 32] for lr in LR_GRID]
        # 4) interaction: hidden x epochs at baseline LR/batch
        s4 = [(h, e, BASE_LR, BASE_BATCH) for h in [4, 16, 32] for e in EPOCHS_GRID]
        # 5) interaction: hidden x batch
        s5 = [(h, BASE_EPOCHS, BASE_LR, b) for h in [4, 16, 32] for b in BATCH_GRID]
        combos = list({c for c in s1 + s2 + s3 + s4 + s5})  # dedup via set
        combos = sorted(combos)
    return combos

COMBOS = build_grid()
print(f"Walk-forward selection 2022-2024 only; 2025 NEVER used")
print(f"Mode={MODE} | combos={len(COMBOS)} | hidden={HIDDEN_GRID} epochs={EPOCHS_GRID} lrs={LR_GRID} batches={BATCH_GRID} seeds={N_SEEDS} folds={FOLDS}")
print(f"Baseline hidden sweep: epochs={BASE_EPOCHS} lr={BASE_LR} batch={BASE_BATCH}")
if MODE != 'hidden_only':
    print(f"Ablation logic: hidden=4 optimal could be (a) underfit/training-limited or (b) data-limited/small-sample bias-variance.")
    print(f"  -> s2 tests (a): does h=4 improve with more epochs/lower LR/smaller batch?")
    print(f"  -> s3-s5 test (b): does larger hidden catch up when given same budget? Interaction hidden x LR/epochs/batch.\n")

rows = []
for cluster in [1, 2, 3]:
    cdata = clean_agg[clean_agg['cluster'] == cluster]
    cdata = cdata[cdata['iso_year'].isin(TRAIN_YEARS)].sort_values('week_start').reset_index(drop=True)
    by_year = {y: cdata[cdata['iso_year'] == y]['weekly_cases'].values for y in TRAIN_YEARS}

    for kind in ['RNN', 'LSTM']:
        for (hidden, epochs, lr, batch) in COMBOS:
            rmses, corrs, biases, train_rmses, best_eps = [], [], [], [], []
            for fold in FOLDS:
                train_vals = np.concatenate([by_year[y] for y in TRAIN_YEARS if y < fold])
                val_vals = by_year[fold]
                if len(val_vals) < T_IN + 2 or len(train_vals) < 20:
                    continue
                n_es = max(int(len(train_vals) * ES_FRAC), 10)
                es_train, es_val = train_vals[:-n_es], train_vals[-n_es:]
                for seed in range(N_SEEDS):
                    model, mu, std, best_ep, tr_rmse = train_model(es_train, es_val, hidden, kind, seed, epochs, lr, batch)
                    if model is None:
                        continue
                    pred = predict(model, val_vals, mu, std)
                    true = val_vals[T_IN:]
                    n = min(len(pred), len(true))
                    if n < 2:
                        continue
                    rmses.append(np.sqrt(np.mean((true[:n] - pred[:n])**2)))
                    corrs.append(np.corrcoef(true[:n], pred[:n])[0, 1] if np.std(pred[:n]) > 0 else 0.0)
                    biases.append(np.mean(pred[:n] - true[:n]))
                    train_rmses.append(tr_rmse)
                    best_eps.append(best_ep)

            if len(rmses) == 0:
                continue
            # param count for compute analysis
            if kind == 'RNN':
                n_params = hidden*1 + hidden*hidden + 2*hidden + hidden*1 + 1
            else:
                n_params = 4*hidden*1 + 4*hidden*hidden + 8*hidden + hidden*1 + 1
            rows.append({
                'cluster': cluster, 'model': kind, 'hidden': hidden,
                'epochs': epochs, 'lr': lr, 'batch': batch,
                'n_params': n_params,
                'val_rmse_mean': np.mean(rmses), 'val_rmse_std': np.std(rmses),
                'val_corr_mean': np.nanmean(corrs),
                'val_bias_mean': np.mean(biases),
                'train_rmse_mean': np.nanmean(train_rmses),
                'best_epoch_mean': np.mean(best_eps) if best_eps else np.nan,
                'n_runs': len(rmses),
            })

sel = pd.DataFrame(rows)
if sel.empty:
    raise RuntimeError("No valid runs - check data / grids")

sel.to_csv(f'{MODEL_OUT}/model_selection_walkforward.csv', index=False)
print(f"\nSaved: {MODEL_OUT}/model_selection_walkforward.csv ({len(sel)} rows, {len(COMBOS)} combos x 2 models x 3 clusters)")

# ---- RESULTS: hidden at baseline (comparable to original) ---------------------
print("\n[1] Hidden sweep at baseline (epochs=150, lr=0.001, batch=16) -- original protocol")
baseline = sel[(sel['epochs'] == BASE_EPOCHS) & (sel['lr'] == BASE_LR) & (sel['batch'] == BASE_BATCH)]
if not baseline.empty:
    pivot = baseline.pivot_table(index=['model', 'hidden'], values='val_rmse_mean', aggfunc='mean').reset_index()
    for kind in ['RNN', 'LSTM']:
        sub = pivot[pivot['model'] == kind].sort_values('hidden')
        if sub.empty:
            continue
        best = sub.loc[sub['val_rmse_mean'].idxmin()]
        print(f"  {kind}: best hidden = {int(best['hidden'])} (RMSE {best['val_rmse_mean']:.1f})")
        for _, r in sub.iterrows():
            print(f"      h={int(r['hidden']):3d} RMSE={r['val_rmse_mean']:7.1f}")
    overall = pivot.groupby('hidden')['val_rmse_mean'].mean().reset_index()
    rec = overall.loc[overall['val_rmse_mean'].idxmin()]
    print(f"  Overall baseline: hidden={int(rec['hidden'])} RMSE {rec['val_rmse_mean']:.1f}")

# ---- RESULTS: training sweep at hidden=4 (underfit test) ---------------------
print("\n[2] Training sweep at hidden=4 (does h=4 improve with more budget? underfit test)")
h4 = sel[sel['hidden'] == 4]
if not h4.empty:
    # best config for h=4
    best_h4 = h4.loc[h4['val_rmse_mean'].idxmin()]
    print(f"  Best h=4 config: epochs={int(best_h4['epochs'])} lr={best_h4['lr']} batch={int(best_h4['batch'])} "
          f"-> val RMSE {best_h4['val_rmse_mean']:.1f} (train {best_h4['train_rmse_mean']:.1f}) "
          f"vs baseline h=4 (150/0.001/16) RMSE {baseline[(baseline['hidden']==4)].groupby('model')['val_rmse_mean'].mean().mean():.1f}")
    # marginal effects
    for var in ['epochs', 'lr', 'batch']:
        marg = h4.groupby(var)['val_rmse_mean'].mean().reset_index().sort_values(var)
        print(f"  Marginal {var}: " + " | ".join(f"{r[var]}:{r['val_rmse_mean']:.1f}" for _, r in marg.iterrows()))
    # if best_h4 RMSE < baseline RMSE by >5%, underfit evidence; else not
    base_h4_rmse = baseline[baseline['hidden'] == 4]['val_rmse_mean'].mean()
    gain = (base_h4_rmse - best_h4['val_rmse_mean']) / base_h4_rmse * 100 if base_h4_rmse else 0
    print(f"  Gain from tuning training at h=4: {gain:.1f}% {'(underfit signal if >5%)' if gain>5 else '(no underfit - budget sufficient)'}")

# ---- RESULTS: interaction hidden x other -------------------------------------
print("\n[3] Interaction hidden x LR (epochs=150, batch=16)")
inter_lr = sel[(sel['epochs'] == BASE_EPOCHS) & (sel['batch'] == BASE_BATCH)]
if not inter_lr.empty:
    piv = inter_lr.pivot_table(index='hidden', columns='lr', values='val_rmse_mean', aggfunc='mean')
    print(piv.round(1).to_string())
    # does larger hidden win at any LR?
    for lr in LR_GRID:
        sub = inter_lr[inter_lr['lr'] == lr].groupby('hidden')['val_rmse_mean'].mean()
        if not sub.empty:
            print(f"  lr={lr}: best h={int(sub.idxmin())} RMSE {sub.min():.1f}")

print("\n[4] Interaction hidden x epochs (lr=0.001, batch=16)")
inter_ep = sel[(sel['lr'] == BASE_LR) & (sel['batch'] == BASE_BATCH)]
if not inter_ep.empty:
    piv = inter_ep.pivot_table(index='hidden', columns='epochs', values='val_rmse_mean', aggfunc='mean')
    print(piv.round(1).to_string())
    for e in EPOCHS_GRID:
        sub = inter_ep[inter_ep['epochs'] == e].groupby('hidden')['val_rmse_mean'].mean()
        if not sub.empty:
            print(f"  epochs={e}: best h={int(sub.idxmin())} RMSE {sub.min():.1f}")

print("\n[5] Full best config (any hidden/epochs/LR/batch)")
overall_best = sel.loc[sel['val_rmse_mean'].idxmin()]
print(f"  Best overall: {overall_best['model']} h={int(overall_best['hidden'])} epochs={int(overall_best['epochs'])} "
      f"lr={overall_best['lr']} batch={int(overall_best['batch'])} cluster={int(overall_best['cluster'])} "
      f"-> val RMSE {overall_best['val_rmse_mean']:.1f} train {overall_best['train_rmse_mean']:.1f} "
      f"n_params {int(overall_best['n_params'])} best_epoch {overall_best['best_epoch_mean']:.0f}")
# best per hidden (to see if larger hidden closes gap when tuned)
best_per_h = sel.groupby('hidden')['val_rmse_mean'].mean().reset_index().sort_values('val_rmse_mean')
print("  Best per hidden (tuned training): " + " | ".join(f"h={int(r['hidden'])}:{r['val_rmse_mean']:.1f}" for _, r in best_per_h.iterrows()))
if best_per_h.iloc[0]['hidden'] != 4:
    print("  -> Larger hidden wins when training tuned: supports underfit hypothesis for h=4")
else:
    print("  -> Hidden=4 still best even when all configs tuned: data-limited, not underfit (small n~52 weeks/yr cannot support >33 params)")

# ---- UNDERFIT DIAGNOSTIC -----------------------------------------------------
# train vs val gap: underfit = both high, overfit = train low val high
print("\n[6] Underfit vs overfit diagnostic (train vs val RMSE gap)")
diag = sel.groupby(['hidden', 'model'])[['train_rmse_mean', 'val_rmse_mean']].mean().reset_index()
for _, r in diag.iterrows():
    gap = r['val_rmse_mean'] - r['train_rmse_mean']
    print(f"  {r['model']} h={int(r['hidden']):2d}: train {r['train_rmse_mean']:.1f} val {r['val_rmse_mean']:.1f} gap {gap:.1f} "
          f"{'underfit' if r['train_rmse_mean']>200 and gap<50 else 'overfit' if gap>100 else 'balanced'}")

# ---- SUMMARY CSV ------------------------------------------------------------
summary_rows = []
# marginal summary
for var in ['hidden', 'epochs', 'lr', 'batch']:
    for v, g in sel.groupby(var):
        summary_rows.append({'factor': var, 'value': v, 'val_rmse_mean': g['val_rmse_mean'].mean(), 'n': len(g)})
pd.DataFrame(summary_rows).to_csv(f'{MODEL_OUT}/model_selection_summary.csv', index=False)
print(f"\nSaved: {MODEL_OUT}/model_selection_summary.csv")

# original overall recommendation
pivot_all = sel.pivot_table(index=['model', 'hidden'], values='val_rmse_mean', aggfunc='mean').reset_index()
overall_all = pivot_all.groupby('hidden')['val_rmse_mean'].mean().reset_index()
rec_all = overall_all.loc[overall_all['val_rmse_mean'].idxmin()]
print(f"\nOverall (tuned) recommendation: hidden={int(rec_all['hidden'])} RMSE {rec_all['val_rmse_mean']:.1f}")
print("Recommendation: use this hidden with its best (epochs,lr,batch) from [5]. If h=4 still best, keep it; else update 06_model_training defaults.")

# ---- FIGURES ----------------------------------------------------------------
# Fig1: original hidden sweep at baseline
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, kind in zip(axes, ['RNN', 'LSTM']):
    sub = baseline[baseline['model'] == kind] if not baseline.empty else sel[sel['model'] == kind]
    for c in [1, 2, 3]:
        d = sub[sub['cluster'] == c].sort_values('hidden')
        if d.empty:
            continue
        ax.plot(d['hidden'], d['val_rmse_mean'], marker='o', label=f'c{c}', lw=1.5)
    ax.set_xlabel('hidden'); ax.set_ylabel('val RMSE')
    ax.set_title(f'{kind} hidden sweep (baseline train)')
    ax.legend()
fig.suptitle('Hidden selection - baseline (no 2025 leakage)', y=0.98)
fig.tight_layout(rect=[0,0,1,0.95])
fig.savefig(os.path.join(FIG_DIR, 'fig_model_selection.png'), dpi=160)
plt.close(fig)
print(f"Saved: {FIG_DIR}/fig_model_selection.png")

# Fig2: hidden x LR interaction
try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for lr in LR_GRID:
        sub = inter_lr[inter_lr['lr'] == lr].groupby('hidden')['val_rmse_mean'].mean().reset_index().sort_values('hidden')
        ax.plot(sub['hidden'], sub['val_rmse_mean'], marker='o', label=f'lr={lr}')
    ax.set_xlabel('hidden'); ax.set_ylabel('mean val RMSE'); ax.set_title('Interaction: hidden x LR (epochs=150,batch=16)')
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig_interaction_hidden_lr.png'), dpi=160); plt.close(fig)
    print(f"Saved: {FIG_DIR}/fig_interaction_hidden_lr.png")
except Exception as e:
    print(f"Skip fig_interaction_hidden_lr: {e}")

# Fig3: hidden x epochs
try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for e in EPOCHS_GRID:
        sub = inter_ep[inter_ep['epochs'] == e].groupby('hidden')['val_rmse_mean'].mean().reset_index().sort_values('hidden')
        ax.plot(sub['hidden'], sub['val_rmse_mean'], marker='o', label=f'epochs={e}')
    ax.set_xlabel('hidden'); ax.set_ylabel('mean val RMSE'); ax.set_title('Interaction: hidden x epochs (lr=0.001,batch=16)')
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig_interaction_hidden_epochs.png'), dpi=160); plt.close(fig)
    print(f"Saved: {FIG_DIR}/fig_interaction_hidden_epochs.png")
except Exception as e:
    print(f"Skip fig_interaction_hidden_epochs: {e}")

# Fig4: underfit diagnostic train vs val
try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for kind in ['RNN', 'LSTM']:
        sub = diag[diag['model'] == kind].sort_values('hidden')
        ax.plot(sub['hidden'], sub['train_rmse_mean'], marker='s', ls='--', label=f'{kind} train')
        ax.plot(sub['hidden'], sub['val_rmse_mean'], marker='o', label=f'{kind} val')
    ax.set_xlabel('hidden'); ax.set_ylabel('RMSE'); ax.set_title('Underfit diagnostic: train vs val RMSE')
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(FIG_DIR, 'fig_underfit_diagnostic.png'), dpi=160); plt.close(fig)
    print(f"Saved: {FIG_DIR}/fig_underfit_diagnostic.png")
except Exception as e:
    print(f"Skip fig_underfit_diagnostic: {e}")

print("\n[OK] Model selection (expanded) complete.")
print("  To test full factorial: python 05_model_selection.py --mode full --hidden 4 8 16 32 --epochs 50 100 150 200 --lrs 0.0005 0.001 0.005 --batches 8 16 32")
print("  To reproduce original: python 05_model_selection.py --mode hidden_only")
