"""§3.1 Harmonic prediction accuracy — reproduces all numbers from the paper.

Claims verified:
  - Ridge R² = 0.86 ± 0.03, GB = 0.88 ± 0.03, MLP = 0.89 ± 0.03 (14 features, 749 materials); the ten assignments' Ridge means within 0.01 of each other
  - Ablation: velocities → 0.61, +Debye → 0.64, +spectral → 0.79, +structural → 0.84, +W → 0.86
  - W lift = +0.02 when structural features present
  - 31 features: Ridge = 0.89, MLP = 0.91, MLP gain ≈ 0.02 over the 14-feature MLP
  - Learning curve (Ridge): gain < 0.01 from ≈ 400 to the full training fold; gradient boosting's curve (same design) is in test_robustness.py

Usage: python test_ceiling.py

RAW RESULTS (749 materials, 14 features; every cross-validated number the mean over ten 10-fold assignments, seeds 0–9)

R2_CEILING Ridge: 0.8601 ± 0.0345  (paper: 0.86 ± 0.03; assignments 0.8558 … 0.8637)  PASS
R2_CEILING GB   : 0.8805 ± 0.0279  (paper: 0.88 ± 0.03; assignments 0.8775 … 0.8833)  PASS
R2_CEILING MLP  : 0.8867 ± 0.0306  (paper: 0.89 ± 0.03; assignments 0.8802 … 0.8930)  PASS
ABLATION velocities    : R² = 0.6119  (paper: 0.61)  PASS
ABLATION +Debye        : R² = 0.6438  (paper: 0.64)  PASS
ABLATION +spectral     : R² = 0.7936  (paper: 0.79)  PASS
ABLATION +structural   : R² = 0.8398  (paper: 0.84)  PASS
ABLATION +W (14 feat)  : R² = 0.8601  (paper: 0.86)  PASS
W_LIFT:          +0.020  (paper: +0.02)  PASS
ROBUSTNESS Ridge(31feat): R² = 0.8916  (paper: 0.89 on 31)  PASS
ROBUSTNESS MLP 31: R² = 0.9060, gain over 14 +0.019  (paper: 0.91, ≈ 0.02)  PASS
LEARNING_CURVE Ridge: gain 404→674 +0.0026 (over the ten assignments, range -0.00002 … +0.00528)  (paper: gain < 0.01)  PASS
OVERALL: PASS  (12/12)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from sklearn.model_selection import cross_val_score
from _data import load_data, make_ridge, make_gb, make_mlp, rcv_r2, FEATURES_14, at_printed


def main():
    from _data import print_stack
    print_stack()
    m, mr = load_data()
    y = m['log_k'].values
    passed = 0
    total = 0

    # ── R² ceiling (paper §3.1, first sentence) ────────────────────────
    print(f'Dataset: {len(m)} materials, 14 features')
    print()

    models = [('Ridge', make_ridge), ('GB', make_gb), ('MLP', make_mlp)]
    paper_r2 = {'Ridge': '0.86', 'GB': '0.88', 'MLP': '0.89'}
    paper_std = {'Ridge': '0.03', 'GB': '0.03', 'MLP': '0.03'}

    for name, model in models:
        X = m[FEATURES_14].values
        r2, std, per = rcv_r2(model, X, y)
        match = at_printed(r2, paper_r2[name]) and at_printed(std, paper_std[name]) and (name != 'Ridge' or max(per) - min(per) < 0.01)   # §2.3: the Ridge means lie within 0.01
        total += 1; passed += match
        print(f'R2_CEILING {name:5s}: {r2:.4f} ± {std:.4f}  '
              f'(paper: {paper_r2[name]} ± {paper_std[name]}; assignments {min(per):.4f} … {max(per):.4f})  '
              f'{"PASS" if match else "FAIL"}')
    print()

    # ── Feature ablation (paper §3.1, ablation paragraph) ──────────────
    ablation_steps = [
        ('velocities',  ['log_sv2pm', 'log_v_LA', 'log_v_TA'], '0.61'),
        ('+Debye',      ['log_sv2pm', 'log_v_LA', 'log_v_TA', 'theta_D_m2_K'], '0.64'),
        ('+spectral',   ['log_sv2pm', 'log_v_LA', 'log_v_TA', 'theta_D_m2_K',
                         'spectral_spread', 'freq_gap_max_THz', 'frac_sv2_below_median',
                         'spectral_entropy', 'dos_cv'], '0.79'),
        ('+structural', ['log_sv2pm', 'log_v_LA', 'log_v_TA', 'theta_D_m2_K',
                         'spectral_spread', 'freq_gap_max_THz', 'frac_sv2_below_median',
                         'spectral_entropy', 'dos_cv',
                         'log_mass', 'delta_en', 'fc_mean', 'interaction_range'], '0.84'),
        ('+W (14 feat)', FEATURES_14, '0.86'),
    ]

    for name, feats, paper_val in ablation_steps:
        r2 = rcv_r2(make_ridge, m[feats].values, y)[0]
        match = at_printed(r2, paper_val)
        total += 1; passed += match
        print(f'ABLATION {name:14s}: R² = {r2:.4f}  (paper: {paper_val})  '
              f'{"PASS" if match else "FAIL"}')

    # W lift
    feats_no_w = [f for f in FEATURES_14 if f != 'log_w']
    r2_no_w = rcv_r2(make_ridge, m[feats_no_w].values, y)[0]
    r2_with_w = rcv_r2(make_ridge, m[FEATURES_14].values, y)[0]
    w_lift = r2_with_w - r2_no_w
    match = at_printed(w_lift, '0.02')
    total += 1; passed += match
    print(f'W_LIFT:          +{w_lift:.3f}  (paper: +0.02)  {"PASS" if match else "FAIL"}')
    print()

    # ── 31-feature robustness check (paper §3.1, robustness paragraph) ─
    extra_cols = []
    from _data import DATA_DIR
    import pandas as pd

    morse_path = os.path.join(DATA_DIR, 'features_morse_analytical_v0.csv')
    proxy_path = os.path.join(DATA_DIR, 'features_anharmonic_proxy_v1.csv')
    for pth in (morse_path, proxy_path):
        if not os.path.exists(pth):
            raise SystemExit(f'ROBUSTNESS: {pth} is missing: the 31-descriptor numbers cannot be reproduced without it')
    morse, proxy = pd.read_csv(morse_path), pd.read_csv(proxy_path)
    for df_, name_ in ((morse, 'Morse'), (proxy, 'proxy')):
        df_['mp_id'] = df_['mp_id'].astype(int)
        assert df_['mp_id'].is_unique, f'ROBUSTNESS: the {name_} CSV has a duplicate mp_id'
    m31 = m.merge(morse, on='mp_id', how='left').merge(proxy, on='mp_id', how='left', suffixes=('', '_proxy'))
    for df_ in (morse, proxy):
        extra_cols.extend(c for c in df_.columns if c not in ('mp_id', 'material') and c in m31.columns and m31[c].dtype in ('float64', 'int64', 'float32'))

    cs_dummies = pd.get_dummies(m31['crystal_system'], prefix='cs')
    for c in cs_dummies.columns:
        m31[c] = cs_dummies[c]
        extra_cols.append(c)

    feats_31 = FEATURES_14 + [c for c in extra_cols if c not in FEATURES_14]
    feats_31 = [c for c in feats_31 if c in m31.columns]
    m31 = m31.dropna(subset=feats_31)
    assert len(m31) == len(m) == 749, f'ROBUSTNESS: the 31-descriptor set has {len(m31)} rows, not the 749'

    X31 = m31[feats_31].values
    y31 = m31['log_k'].values
    r2_ridge_31 = rcv_r2(make_ridge, X31, y31)[0]
    r2_mlp_14 = rcv_r2(make_mlp, m31[FEATURES_14].values, y31)[0]
    r2_mlp_31 = rcv_r2(make_mlp, X31, y31)[0]
    mlp_gain = r2_mlp_31 - r2_mlp_14

    match_r = at_printed(r2_ridge_31, '0.89') and len(feats_31) == 31
    total += 1; passed += int(match_r)
    print(f'ROBUSTNESS Ridge({len(feats_31)}feat): R² = {r2_ridge_31:.4f}  (paper: 0.89 on 31)  '
          f'{"PASS" if match_r else "FAIL"}')

    match_m = at_printed(mlp_gain, '0.02') and at_printed(r2_mlp_31, '0.91')            # the text: the network gaining ≈ 0.02
    total += 1; passed += int(match_m)
    print(f'ROBUSTNESS MLP 31: R² = {r2_mlp_31:.4f}, gain over 14 {mlp_gain:+.3f}  (paper: 0.91, ≈ 0.02)  '
          f'{"PASS" if match_m else "FAIL"}')
    print()

    # ── Learning curve (paper §3.1, last paragraph): the same design as gradient boosting's in test_robustness.py ──
    # each held-out fold is kept; the model trains on a random 60 % of its training fold (≈ 400 materials) and on all of it
    from sklearn.model_selection import learning_curve
    from _data import SEEDS, kfold
    X_full = m[FEATURES_14].values
    gains = []
    for s_ in SEEDS:
        sizes, _, te = learning_curve(make_ridge(), X_full, y, cv=kfold(s_), train_sizes=[.6, 1.0], scoring='r2', shuffle=True, random_state=s_)
        gains.append(te.mean(1)[1] - te.mean(1)[0])
    gain = float(np.mean(gains))
    saturation = gain < 0.01                                              # the text: gain < 0.01 from N ≈ 400 to the full training fold
    total += 1; passed += saturation
    print(f'LEARNING_CURVE Ridge: gain {sizes[0]}→{sizes[1]} {gain:+.4f} (over the ten assignments, range {min(gains):+.5f} … {max(gains):+.5f})  '
          f'(paper: gain < 0.01)  {"PASS" if saturation else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'  [{time.time()-t0:.1f}s]')
    sys.exit(0 if ok else 1)
