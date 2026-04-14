"""§3.1 Harmonic prediction accuracy — reproduces all numbers from the paper.

Claims verified:
  - Ridge R² = 0.86 ± 0.04, GB = 0.88 ± 0.02, MLP = 0.89 ± 0.03 (14 features, 751 materials)
  - Ablation: velocities → 0.61, +Debye → 0.63, +spectral → 0.79, +structural → 0.83, +W → 0.86
  - W lift = +0.02 when structural features present
  - 33 features: Ridge = 0.89, MLP gain < 0.02 over 14-feature MLP
  - Learning curve largely saturates by N ≈ 400 (gain < 0.01 from 400 to 751)

Usage: python test_ceiling.py

RAW RESULTS (2026-04-14, 751 materials, 14 features, 10-fold CV seed=42)

R2_CEILING Ridge: 0.86 ± 0.04  GB: 0.88 ± 0.02  MLP: 0.89 ± 0.03
ABLATION vel=0.61  +Debye=0.63  +spec=0.79  +struct=0.83  +W=0.86
W_LIFT: +0.023
ROBUSTNESS Ridge(33feat)=0.89  MLP gain=+0.015
LEARNING_CURVE: R²(300)=0.831  R²(400)=0.853  R²(full)=0.860
OVERALL: PASS (12/12)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.model_selection import cross_val_score
from _data import load_data, make_ridge, make_gb, make_mlp, cv10, FEATURES_14


def main():
    m, mr = load_data()
    cv = cv10()
    y = m['log_k'].values
    passed = 0
    total = 0

    # ── R² ceiling (paper §3.1, first sentence) ────────────────────────
    print(f'Dataset: {len(m)} materials, 14 features')
    print()

    models = [('Ridge', make_ridge()), ('GB', make_gb()), ('MLP', make_mlp())]
    paper_r2 = {'Ridge': 0.86, 'GB': 0.88, 'MLP': 0.89}
    paper_std = {'Ridge': 0.04, 'GB': 0.02, 'MLP': 0.03}
    tol_r2 = {'Ridge': 0.015, 'GB': 0.015, 'MLP': 0.015}
    tol_std = {'Ridge': 0.015, 'GB': 0.015, 'MLP': 0.015}

    for name, model in models:
        X = m[FEATURES_14].values
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        r2, std = scores.mean(), scores.std()
        match = abs(r2 - paper_r2[name]) < tol_r2[name] and abs(std - paper_std[name]) < tol_std[name]
        total += 1; passed += match
        print(f'R2_CEILING {name:5s}: {r2:.2f} ± {std:.2f}  '
              f'(paper: {paper_r2[name]:.2f} ± {paper_std[name]:.2f})  '
              f'{"PASS" if match else "FAIL"}')
    print()

    # ── Feature ablation (paper §3.1, ablation paragraph) ──────────────
    ablation_steps = [
        ('velocities',  ['log_sv2pm', 'log_v_LA', 'log_v_TA'], 0.61),
        ('+Debye',      ['log_sv2pm', 'log_v_LA', 'log_v_TA', 'theta_D_m2_K'], 0.63),
        ('+spectral',   ['log_sv2pm', 'log_v_LA', 'log_v_TA', 'theta_D_m2_K',
                         'spectral_spread', 'freq_gap_max_THz', 'frac_sv2_below_median',
                         'spectral_entropy', 'dos_cv'], 0.79),
        ('+structural', ['log_sv2pm', 'log_v_LA', 'log_v_TA', 'theta_D_m2_K',
                         'spectral_spread', 'freq_gap_max_THz', 'frac_sv2_below_median',
                         'spectral_entropy', 'dos_cv',
                         'log_mass', 'delta_en', 'fc_mean', 'interaction_range'], 0.83),
        ('+W (14 feat)', FEATURES_14, 0.86),
    ]

    for name, feats, paper_val in ablation_steps:
        X = m[feats].values
        scores = cross_val_score(make_ridge(), X, y, cv=cv, scoring='r2')
        r2 = scores.mean()
        match = abs(r2 - paper_val) < 0.015
        total += 1; passed += match
        print(f'ABLATION {name:14s}: R² = {r2:.2f}  (paper: {paper_val:.2f})  '
              f'{"PASS" if match else "FAIL"}')

    # W lift
    feats_no_w = [f for f in FEATURES_14 if f != 'log_w']
    r2_no_w = cross_val_score(make_ridge(), m[feats_no_w].values, y, cv=cv, scoring='r2').mean()
    r2_with_w = cross_val_score(make_ridge(), m[FEATURES_14].values, y, cv=cv, scoring='r2').mean()
    w_lift = r2_with_w - r2_no_w
    match = abs(w_lift - 0.02) < 0.01
    total += 1; passed += match
    print(f'W_LIFT:          +{w_lift:.3f}  (paper: +0.02)  {"PASS" if match else "FAIL"}')
    print()

    # ── 36-feature robustness check (paper §3.1, robustness paragraph) ─
    extra_cols = []
    from _data import DATA_DIR
    import pandas as pd

    morse_path = os.path.join(DATA_DIR, 'features_morse_analytical_v0.csv')
    if os.path.exists(morse_path):
        morse = pd.read_csv(morse_path)
        morse['mp_id'] = morse['mp_id'].astype(int)
        m36 = m.merge(morse, on='mp_id', how='left')
        morse_feats = [c for c in morse.columns
                       if c not in ('mp_id', 'material') and c in m36.columns
                       and m36[c].dtype in ('float64', 'int64', 'float32')]
        extra_cols.extend(morse_feats)
    else:
        m36 = m.copy()

    proxy_path = os.path.join(DATA_DIR, 'features_anharmonic_proxy_v1.csv')
    if os.path.exists(proxy_path):
        proxy = pd.read_csv(proxy_path)
        proxy['mp_id'] = proxy['mp_id'].astype(int)
        m36 = m36.merge(proxy, on='mp_id', how='left')
        proxy_feats = [c for c in proxy.columns
                       if c not in ('mp_id', 'material') and c in m36.columns
                       and m36[c].dtype in ('float64', 'int64', 'float32')]
        extra_cols.extend(proxy_feats)

    cs_dummies = pd.get_dummies(m36['crystal_system'], prefix='cs')
    for c in cs_dummies.columns:
        m36[c] = cs_dummies[c]
        extra_cols.append(c)

    feats_36 = FEATURES_14 + [c for c in extra_cols if c not in FEATURES_14]
    feats_36 = [c for c in feats_36 if c in m36.columns]
    m36 = m36.dropna(subset=feats_36)

    X36 = m36[feats_36].values
    y36 = m36['log_k'].values
    cv36 = cv10()

    r2_ridge_36 = cross_val_score(make_ridge(), X36, y36, cv=cv36, scoring='r2').mean()
    r2_mlp_14 = cross_val_score(make_mlp(), m36[FEATURES_14].values, y36, cv=cv36, scoring='r2').mean()
    r2_mlp_36 = cross_val_score(make_mlp(), X36, y36, cv=cv36, scoring='r2').mean()
    mlp_gain = r2_mlp_36 - r2_mlp_14

    match_r = abs(r2_ridge_36 - 0.89) < 0.02
    total += 1; passed += int(match_r)
    print(f'ROBUSTNESS Ridge({len(feats_36)}feat): R² = {r2_ridge_36:.2f}  (paper: 0.89)  '
          f'{"PASS" if match_r else "FAIL"}')

    match_m = mlp_gain < 0.02
    total += 1; passed += int(match_m)
    print(f'ROBUSTNESS MLP gain {len(feats_36)} vs 14: {mlp_gain:+.3f}  (paper: <0.02)  '
          f'{"PASS" if match_m else "FAIL"}')
    print()

    # ── Learning curve (paper §3.1, last paragraph) ────────────────────
    X_full = m[FEATURES_14].values
    from sklearn.model_selection import KFold

    r2_at = {300: [], 400: [], len(m): []}
    for seed in range(5):
        rng = np.random.RandomState(seed)
        for n in r2_at:
            idx = rng.choice(len(m), size=min(n, len(m)), replace=False)
            cv_lc = KFold(10, shuffle=True, random_state=42)
            s = cross_val_score(make_ridge(), X_full[idx], y[idx], cv=cv_lc, scoring='r2')
            r2_at[n].append(s.mean())

    diff_300 = abs(np.mean(r2_at[len(m)]) - np.mean(r2_at[300]))
    diff_400 = abs(np.mean(r2_at[len(m)]) - np.mean(r2_at[400]))
    saturation = diff_300 < 0.05 and diff_400 < 0.02
    total += 1; passed += saturation
    print(f'LEARNING_CURVE: R²(300)={np.mean(r2_at[300]):.3f}  '
          f'R²(400)={np.mean(r2_at[400]):.3f}  '
          f'R²(full)={np.mean(r2_at[len(m)]):.3f}  '
          f'diff(300)={diff_300:.3f}  diff(400)={diff_400:.3f}  '
          f'(paper: largely saturates by 300-400)  {"PASS" if saturation else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'  [{time.time()-t0:.1f}s]')
    sys.exit(0 if ok else 1)
