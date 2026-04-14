"""
Generate all figures for the paper.

Figures:
  fig1_ceiling.pdf       — predicted vs actual log(κ), Ridge/GB/MLP
  fig2_residual.pdf      — harmonic residual vs scattering rate Γ
  fig3_stiffness.pdf     — fc_mean vs |V₃|², colored by mass
  fig4_outliers.pdf      — residual plot with rigid oxide/layered highlighted
  fig5_learning.pdf      — R² vs training set size (learning curve)

Usage:
  python make_figures.py          # all figures
  python make_figures.py fig2     # single figure

Data: reads CSVs from ../data/ (standalone, no st_3 dependency)
"""
import sys
import os
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.normpath(os.path.join(HERE, '..', 'data'))

from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_predict, cross_val_score

plt.rcParams.update({
    'font.size': 9, 'axes.labelsize': 10, 'xtick.labelsize': 8,
    'ytick.labelsize': 8, 'legend.fontsize': 8, 'figure.dpi': 300,
})

BTE_ARTIFACTS = {1105, 9497, 4823}

EN_PAULING = {
    'H':2.20,'Li':0.98,'Be':1.57,'B':2.04,'C':2.55,'N':3.04,'O':3.44,'F':3.98,
    'Na':0.93,'Mg':1.31,'Al':1.61,'Si':1.90,'P':2.19,'S':2.58,'Cl':3.16,
    'K':0.82,'Ca':1.00,'Sc':1.36,'Ti':1.54,'V':1.63,'Cr':1.66,'Mn':1.55,'Fe':1.83,
    'Co':1.88,'Ni':1.91,'Cu':1.90,'Zn':1.65,'Ga':1.81,'Ge':2.01,'As':2.18,'Se':2.55,'Br':2.96,
    'Rb':0.82,'Sr':0.95,'Y':1.22,'Zr':1.33,'Nb':1.60,'Mo':2.16,'Ru':2.20,'Rh':2.28,
    'Pd':2.20,'Ag':1.93,'Cd':1.69,'In':1.78,'Sn':1.96,'Sb':2.05,'Te':2.10,'I':2.66,
    'Cs':0.79,'Ba':0.89,'La':1.10,'Hf':1.30,'Ta':1.50,'W':2.36,'Re':1.90,'Os':2.20,
    'Ir':2.20,'Pt':2.28,'Au':2.54,'Tl':1.62,'Pb':2.33,'Bi':2.02,
    'Ce':1.12,'Pr':1.13,'Nd':1.14,'Sm':1.17,'Eu':1.20,'Gd':1.20,'Tb':1.10,
    'Dy':1.22,'Ho':1.23,'Er':1.24,'Tm':1.25,'Lu':1.27,'Th':1.30,'U':1.38,
}

import re

def compute_delta_en(formula):
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    elements = []
    for el, n in tokens:
        if el and el in EN_PAULING:
            count = int(n) if n else 1
            elements.extend([EN_PAULING[el]] * count)
    if len(elements) < 2:
        return np.nan
    return max(elements) - min(elements)


def load_data():
    paths = {
        'mesh': 'features_mesh_v1.csv', 'fc': 'features_fc_v1.csv',
        'debye': 'debye_v1.csv', 'sv2': 'features_sv2_v1.csv',
        'spectral': 'features_spectral_v1.csv', 'sg': 'spacegroup_phonopy_v1.csv',
        'ps': 'features_phase_space_v1.csv',
    }
    dfs = {}
    for key, fname in paths.items():
        dfs[key] = pd.read_csv(os.path.join(DATA, fname))
        dfs[key]['mp_id'] = dfs[key]['mp_id'].astype(int)

    kb = pd.read_csv(os.path.join(DATA, 'kappa_bte_all.csv'))
    kb['mp_id'] = kb['mp_id'].astype(int)

    m = dfs['mesh'][['mp_id', 'material', 'v_LA_m_per_s', 'v_TA_m_per_s',
                      'n_atoms_primitive', 'sound_velocity_unresolved',
                      'dos_width_THz', 'freq_max_THz']].copy()
    m = m.merge(dfs['sv2'][['mp_id', 'sv2_per_mode']], on='mp_id')
    m = m.merge(dfs['fc'][['mp_id', 'avg_mass_AMU', 'fc_mean', 'interaction_range']], on='mp_id')
    m = m.merge(dfs['debye'][['mp_id', 'theta_D_m2_K']], on='mp_id')
    m = m.merge(dfs['spectral'][['mp_id', 'spectral_spread', 'freq_gap_max_THz',
                                  'frac_sv2_below_median', 'spectral_entropy']], on='mp_id')
    m = m.merge(kb[['mp_id', 'k_bte_avg']], on='mp_id')
    m = m.merge(dfs['sg'][['mp_id', 'crystal_system']], on='mp_id')
    m = m.merge(dfs['ps'][['mp_id', 'w_mean']], on='mp_id', how='left')

    m['sound_velocity_unresolved'] = m['sound_velocity_unresolved'].astype(bool)
    m = m[~m.sound_velocity_unresolved]
    m = m.dropna(subset=['k_bte_avg'])
    m = m[~m.mp_id.isin(BTE_ARTIFACTS)]

    m['log_k'] = np.log10(m['k_bte_avg'])
    m['log_v_LA'] = np.log10(m['v_LA_m_per_s'])
    m['log_v_TA'] = np.log10(m['v_TA_m_per_s'])
    m['log_mass'] = np.log10(m['avg_mass_AMU'])
    m['log_sv2pm'] = np.log10(m['sv2_per_mode'])
    m['delta_en'] = m['material'].apply(compute_delta_en)
    m['dos_cv'] = m['dos_width_THz'] / m['freq_max_THz']
    m['log_w'] = np.log10(m['w_mean'].clip(lower=1e-8))
    m = m.dropna(subset=['delta_en'])

    rta_path = os.path.join(DATA, 'features_scattering_rta_v1.csv')
    rta = None
    if os.path.exists(rta_path):
        rta = pd.read_csv(rta_path)
        rta['mp_id'] = rta['mp_id'].astype(int)
        rta = rta[rta.kappa_rta < 1000].copy()
        rta['log_krta'] = np.log10(rta['kappa_rta'].clip(lower=0.01))
        rta['log_gamma'] = np.log10(rta['gamma_transport'].clip(lower=1e-8))
        rta['log_V3sq'] = rta['log_gamma'] - np.log10(rta['w_mean_rta'].clip(lower=1e-8)) if 'w_mean_rta' in rta.columns else rta['log_gamma']

    return m, rta


def _features(m):
    cols = ['log_v_LA', 'log_v_TA', 'log_sv2pm', 'log_mass',
            'theta_D_m2_K', 'spectral_spread', 'freq_gap_max_THz',
            'frac_sv2_below_median', 'spectral_entropy', 'delta_en',
            'dos_cv', 'interaction_range', 'fc_mean']
    if 'log_w' in m.columns:
        cols.append('log_w')
    return [c for c in cols if c in m.columns]


def _ridge():
    return Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=1.0))])


def _gb():
    return GradientBoostingRegressor(n_estimators=500, max_depth=4, random_state=42)


def _mlp():
    return Pipeline([('sc', StandardScaler()),
                     ('m', MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                                        max_iter=1000, random_state=42))])


# ── Figure 1: predicted vs actual ─────────────────────────────────────

def fig1_ceiling(m):
    print("Generating fig1_ceiling.pdf ...")
    feat = _features(m)
    X = m[feat].values
    y = m['log_k'].values
    cv = KFold(10, shuffle=True, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.3), sharex=True, sharey=True)
    for ax, (name, model) in zip(axes, [('Ridge', _ridge()), ('GB', _gb()), ('MLP', _mlp())]):
        pred = cross_val_predict(model, X, y, cv=cv)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        r2_mean = np.mean(scores)
        r2_std = np.std(scores)
        ax.scatter(y, pred, s=6, alpha=0.4, c='steelblue', edgecolors='none')
        ax.plot([-1, 3.5], [-1, 3.5], 'k--', lw=0.8)
        ax.set_title(f'{name}  R² = {r2_mean:.2f} ± {r2_std:.2f}')
        ax.set_xlabel('Actual log₁₀(κ)')
        if ax == axes[0]:
            ax.set_ylabel('Predicted log₁₀(κ)')

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig1_ceiling.pdf'), bbox_inches='tight')
    plt.close()
    print("  done.")


# ── Figure 2: residual vs Γ ───────────────────────────────────────────

def fig2_residual(m, rta):
    print("Generating fig2_residual.pdf ...")
    if rta is None:
        print("  SKIP (no RTA data)")
        return

    feat = _features(m)
    feat_w = [f for f in feat if f != 'log_w'] + ['log_w']
    mr = m.merge(rta[['mp_id', 'log_gamma', 'log_krta']], on='mp_id')

    X = mr[feat_w].values
    y = mr['log_k'].values
    cv = KFold(10, shuffle=True, random_state=42)
    pred = cross_val_predict(_ridge(), X, y, cv=cv)
    resid = y - pred

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(mr['log_gamma'].values, resid, s=12, alpha=0.5,
                    c='steelblue', edgecolors='none')

    from scipy.stats import pearsonr
    r, p = pearsonr(mr['log_gamma'].values, resid)
    ax.set_xlabel('log₁₀(Γ)  [scattering rate]')
    ax.set_ylabel('Harmonic residual  [log₁₀(κ) − predicted]')
    ax.set_title(f'r = {r:.3f}')
    ax.axhline(0, color='gray', lw=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig2_residual.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  done. r={r:.3f}")


# ── Figure 3: fc_mean vs |V₃|² ───────────────────────────────────────

def fig3_stiffness(m, rta):
    print("Generating fig3_stiffness.pdf ...")
    if rta is None:
        print("  SKIP (no RTA data)")
        return

    mr = m.merge(rta[['mp_id', 'log_gamma']], on='mp_id')
    mr['log_V3sq'] = mr['log_gamma'] - mr['log_w']

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(np.log10(mr['fc_mean'].clip(lower=1e-3)),
                    mr['log_V3sq'],
                    c=mr['log_mass'], cmap='viridis', s=12, alpha=0.6,
                    edgecolors='none')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('log₁₀(mass)')

    from scipy.stats import pearsonr
    log_fc = np.log10(mr['fc_mean'].clip(lower=1e-3))
    r_raw, _ = pearsonr(log_fc, mr['log_V3sq'])
    from sklearn.linear_model import LinearRegression
    mass_vals = mr['log_mass'].values.reshape(-1, 1)
    fc_resid = log_fc.values - LinearRegression().fit(mass_vals, log_fc.values).predict(mass_vals)
    v3_resid = mr['log_V3sq'].values - LinearRegression().fit(mass_vals, mr['log_V3sq'].values).predict(mass_vals)
    r_partial, _ = pearsonr(fc_resid, v3_resid)
    ax.set_xlabel('log₁₀(fc_mean)')
    ax.set_ylabel('log₁₀(|V₃|²)')
    ax.set_title(f'r = {r_raw:.3f}  (partial|mass = {r_partial:.2f})')

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig3_stiffness.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  done. r_raw={r_raw:.3f}")


# ── Figure 4: outlier classes ─────────────────────────────────────────

def fig4_outliers(m, rta):
    print("Generating fig4_outliers.pdf ...")
    if rta is None:
        print("  SKIP (no RTA data)")
        return

    feat = _features(m)
    mr = m.merge(rta[['mp_id', 'log_gamma']], on='mp_id')

    X = mr[feat].values
    y = mr['log_k'].values
    cv = KFold(10, shuffle=True, random_state=42)
    pred = cross_val_predict(_ridge(), X, y, cv=cv)
    resid = y - pred

    mr = mr.copy()
    mr['resid'] = resid
    mr['log_V3sq'] = mr['log_gamma'] - mr['log_w']

    perov_names = ['TiO2', 'ReO3', 'ZrO2', 'SnO2']
    layer_names = ['GaSe', 'GaS', 'GaTe', 'WS2', 'MoS2', 'WSe2', 'BiI']

    is_perov = mr['material'].isin(perov_names)
    is_layer = mr['material'].isin(layer_names)
    is_other = ~is_perov & ~is_layer

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(mr.loc[is_other, 'log_V3sq'], mr.loc[is_other, 'resid'],
               s=10, alpha=0.3, c='gray', edgecolors='none', label='other')
    ax.scatter(mr.loc[is_perov, 'log_V3sq'], mr.loc[is_perov, 'resid'],
               s=30, alpha=0.9, c='tomato', edgecolors='k', lw=0.5, label='rigid oxide', zorder=5)
    ax.scatter(mr.loc[is_layer, 'log_V3sq'], mr.loc[is_layer, 'resid'],
               s=30, alpha=0.9, c='royalblue', edgecolors='k', lw=0.5, label='layered', zorder=5)

    for _, row in mr[is_perov | is_layer].iterrows():
        ax.annotate(row['material'], (row['log_V3sq'], row['resid']),
                    fontsize=6, alpha=0.8, xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('log₁₀(|V₃|²)')
    ax.set_ylabel('Harmonic residual')
    ax.axhline(0, color='gray', lw=0.5)
    ax.legend(loc='lower left')

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig4_outliers.pdf'), bbox_inches='tight')
    plt.close()
    print("  done.")


# ── Figure 5: learning curve ─────────────────────────────────────────

def fig5_learning(m):
    print("Generating fig5_learning.pdf ...")
    feat = _features(m)
    X = m[feat].values
    y = m['log_k'].values

    sizes = [50, 100, 200, 300, 400, 500, 600, len(m)]
    results = {n: [] for n in ['Ridge', 'GB', 'MLP']}

    for n in sizes:
        for seed in range(5):
            rng = np.random.RandomState(seed)
            idx = rng.choice(len(m), size=min(n, len(m)), replace=False)
            X_sub, y_sub = X[idx], y[idx]
            cv = KFold(min(5, len(X_sub)), shuffle=True, random_state=42)

            for name, model in [('Ridge', _ridge()), ('GB', _gb()), ('MLP', _mlp())]:
                try:
                    scores = cross_val_score(model, X_sub, y_sub, cv=cv, scoring='r2')
                    results[name].append((n, np.mean(scores)))
                except Exception:
                    results[name].append((n, np.nan))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    colors = {'Ridge': 'steelblue', 'GB': 'darkorange', 'MLP': 'seagreen'}
    for name in ['Ridge', 'GB', 'MLP']:
        pts = [(n, r2) for n, r2 in results[name] if not np.isnan(r2)]
        if not pts:
            continue
        ns, r2s = zip(*pts)
        ns_arr = np.array(ns)
        r2_arr = np.array(r2s)
        unique_n = sorted(set(ns_arr))
        means = [np.mean(r2_arr[ns_arr == n]) for n in unique_n]
        stds = [np.std(r2_arr[ns_arr == n]) for n in unique_n]
        ax.errorbar(unique_n, means, yerr=stds, marker='o', ms=4, lw=1.2,
                    label=name, color=colors[name], capsize=3)

    ax.set_xlabel('Training set size')
    ax.set_ylabel('Cross-validated R²')
    ax.legend()
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(os.path.join(HERE, 'fig5_learning.pdf'), bbox_inches='tight')
    plt.close()
    print("  done.")


# ── Main ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    m, rta = load_data()
    print(f"Loaded {len(m)} materials, RTA subset: {len(rta) if rta is not None else 0}")

    targets = sys.argv[1:] if len(sys.argv) > 1 else ['fig1', 'fig2', 'fig3', 'fig4', 'fig5']

    if 'fig1' in targets:
        fig1_ceiling(m)
    if 'fig2' in targets:
        fig2_residual(m, rta)
    if 'fig3' in targets:
        fig3_stiffness(m, rta)
    if 'fig4' in targets:
        fig4_outliers(m, rta)
    if 'fig5' in targets:
        fig5_learning(m)

    print("\nAll requested figures generated.")
