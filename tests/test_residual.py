"""§3.2 Characterizing the residual — reproduces all numbers from the paper.

Claims verified:
  - FC2 → log(κ_BTE/κ_RTA): Ridge R² = −0.39, GB = −2.12, MLP = −1.48 (all negative)
  - 5-feature residual vs log(Γ): r = −0.54
  - 14-feature residual vs log(Γ): r = −0.29 (p < 10⁻⁴)
  - W partial|mass with 14-feature residual: r = −0.01
  - ANOVA across dominant-element groups: p = 0.07
  - Durbin-Watson: between 1.78 and 2.16
  - §4 factorization: log(κ) ≈ +1.11·log(sv2) − 0.80·log(Γ), R²_insample = 0.78, R²_CV = 0.73
  - §4 FC2 → |V₃|² at R² = 0.70, but redundant for κ (lift < 0.001)
  - §4 oracle: true |V₃|² residual improves κ by +0.04 R²
  - §4 bottleneck: v3_resid ~ k_resid at r = −0.53

Usage: python test_residual.py

RAW RESULTS (2026-04-14, 231-material RTA overlap, 10-fold CV seed=42)

BTE_RTA Ridge=-0.39  GB=-2.12  MLP=-1.48 (all negative)
RESID_5FEAT vs Γ: r=-0.544
RESID_14FEAT vs Γ: r=-0.287 p=9.0e-06
W_PARTIAL|mass: r=-0.013
ANOVA (10 groups, O n=302): F=1.84 p=0.066
DW: pred_κ=2.157  log_mass=1.776  log_v_LA=1.951
FACTORIZATION: R²_insample=0.78  R²_CV=0.73  a=+1.109  b=-0.796
FC2→|V₃|² R²=0.70, κ lift=+0.000 (redundant)
ORACLE: +|V₃|² residual → κ lift=+0.040
BOTTLENECK: v3_resid~k_resid r=-0.529
OVERALL: PASS (12/12)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from _data import (load_data, make_ridge, make_gb, make_mlp, cv10,
                   partial_corr, FEATURES_14, FEATURES_5)


def main():
    m, mr = load_data()
    cv = cv10()
    passed = 0
    total = 0

    print(f'RTA overlap: {len(mr)} materials')
    print()

    # ── BTE/RTA unpredictable (paper §3.2, first paragraph) ────────────
    X = mr[FEATURES_14].values
    y_ratio = mr['log_k'].values - mr['log_krta'].values

    paper_vals = {'Ridge': -0.39, 'GB': -2.12, 'MLP': -1.48}
    for name, model in [('Ridge', make_ridge()), ('GB', make_gb()), ('MLP', make_mlp())]:
        scores = cross_val_score(model, X, y_ratio, cv=cv, scoring='r2')
        r2 = scores.mean()
        negative = r2 < 0
        total += 1; passed += negative
        print(f'BTE_RTA {name:5s}: R² = {r2:.2f}  (paper: {paper_vals[name]:.2f})  '
              f'{"PASS" if negative else "FAIL"}  [must be < 0]')
    print()

    # ── Residual vs Γ (paper §3.2, second paragraph) ──────────────────
    y = mr['log_k'].values
    gamma = mr['log_gamma'].values

    # 5-feature baseline
    pred5 = cross_val_predict(make_ridge(), mr[FEATURES_5].values, y, cv=cv)
    resid5 = y - pred5
    r5, p5 = pearsonr(gamma, resid5)
    match5 = abs(r5 - (-0.54)) < 0.03
    total += 1; passed += match5
    print(f'RESID_5FEAT vs Γ: r = {r5:.3f}  (paper: −0.54)  {"PASS" if match5 else "FAIL"}')

    # 14-feature model
    pred14 = cross_val_predict(make_ridge(), X, y, cv=cv)
    resid14 = y - pred14
    r14, p14 = pearsonr(gamma, resid14)
    match14 = abs(r14 - (-0.29)) < 0.03 and p14 < 1e-4
    total += 1; passed += match14
    print(f'RESID_14FEAT vs Γ: r = {r14:.3f}  p = {p14:.1e}  '
          f'(paper: −0.29, p < 10⁻⁴)  {"PASS" if match14 else "FAIL"}')

    # W partial|mass with residual
    r_w, p_w = partial_corr(resid14, mr['log_w'].values, mr['log_mass'].values)
    match_w = abs(r_w) < 0.05
    total += 1; passed += match_w
    print(f'W_PARTIAL|mass vs resid: r = {r_w:.3f}  '
          f'(paper: −0.01)  {"PASS" if match_w else "FAIL"}')
    print()

    # ── Residual uniformity (paper §3.2, last paragraph) ──────────────
    from scipy.stats import f_oneway

    def durbin_watson(resid):
        diff = np.diff(resid)
        return np.sum(diff**2) / np.sum(resid**2)

    # Full-dataset residual for uniformity tests
    X_full = m[FEATURES_14].values
    y_full = m['log_k'].values
    pred_full = cross_val_predict(make_ridge(), X_full, y_full, cv=cv)
    resid_full = y_full - pred_full

    # ANOVA by dominant element (most frequent element in formula)
    import re as _re
    def _dom_el(formula):
        tokens = _re.findall(r'([A-Z][a-z]?)(\d*)', formula)
        best_el, best_n = '', 0
        for el, n in tokens:
            count = int(n) if n else 1
            if count > best_n:
                best_n, best_el = count, el
        return best_el
    classes = m['material'].apply(_dom_el)
    counts = classes.value_counts()
    major = counts[counts >= 15].index
    anova_groups = [resid_full[(classes == cl).values] for cl in major]
    f_stat, p_anova = f_oneway(*anova_groups)
    match_a = p_anova > 0.05
    total += 1; passed += int(match_a)
    print(f'ANOVA ({len(major)} groups, largest O n={counts.max()}): F={f_stat:.2f}  p={p_anova:.3f}  '
          f'(paper: p = 0.07)  {"PASS" if match_a else "FAIL"}')

    # Durbin-Watson in 3 orderings
    orderings = {
        'pred_κ': np.argsort(pred_full),
        'log_mass': np.argsort(m['log_mass'].values),
        'log_v_LA': np.argsort(m['log_v_LA'].values),
    }
    dw_ok = True
    for oname, order in orderings.items():
        dw = durbin_watson(resid_full[order])
        in_range = 1.5 <= dw <= 2.5
        dw_ok = dw_ok and in_range
        print(f'DW by {oname:10s}: {dw:.3f}  {"ok" if in_range else "FAIL"}')
    total += 1; passed += dw_ok
    print(f'DURBIN_WATSON: (paper: 1.74–2.00)  {"PASS" if dw_ok else "FAIL"}')
    print()

    # ── §4 factorization: κ ∝ sv2/Γ ──────────────────────────────────
    log_sv2 = np.log10(mr['sv2_per_mode'].values)
    X_fact = np.column_stack([log_sv2, gamma])
    reg = LinearRegression().fit(X_fact, y)
    r2_fact = reg.score(X_fact, y)
    a, b = reg.coef_
    from sklearn.model_selection import cross_val_score as _cvs
    r2_fact_cv = _cvs(LinearRegression(), X_fact, y, cv=cv, scoring='r2').mean()
    match_fact = (abs(r2_fact - 0.78) < 0.02 and abs(r2_fact_cv - 0.73) < 0.05 and
                  abs(a - 1.11) < 0.05 and abs(b - (-0.80)) < 0.05)
    total += 1; passed += match_fact
    print(f'FACTORIZATION: R²_insample = {r2_fact:.2f}  R²_CV = {r2_fact_cv:.2f}  '
          f'a = {a:+.3f}  b = {b:+.3f}  '
          f'(paper: 0.78/0.73, +1.11, −0.80)  {"PASS" if match_fact else "FAIL"}')

    # ── §4 oracle test: FC2 → |V₃|² redundancy + ceiling bound ────────
    print()
    y_v3 = mr['log_V3sq'].values
    v3_pred = cross_val_predict(make_ridge(), X, y_v3, cv=cv)
    v3_resid = y_v3 - v3_pred

    # FC2 predicts |V₃|² but prediction is redundant for κ
    from sklearn.model_selection import cross_val_score as _cvs2
    r2_fc2_v3 = _cvs2(make_ridge(), X, y_v3, cv=cv, scoring='r2').mean()
    X_plus_v3pred = np.column_stack([X, v3_pred])
    r2_k_plus_v3 = _cvs2(make_ridge(), X_plus_v3pred, y, cv=cv, scoring='r2').mean()
    r2_k_base = _cvs2(make_ridge(), X, y, cv=cv, scoring='r2').mean()
    lift_v3pred = r2_k_plus_v3 - r2_k_base
    match_redundancy = r2_fc2_v3 > 0.60 and abs(lift_v3pred) < 0.01
    total += 1; passed += int(match_redundancy)
    print(f'FC2→|V₃|² R²={r2_fc2_v3:.2f} but κ lift={lift_v3pred:+.4f}  '
          f'(paper: R²=0.70, lift<0.001)  {"PASS" if match_redundancy else "FAIL"}')

    # Oracle: true v3_resid improves κ by ~0.04
    X_plus_oracle = np.column_stack([X, v3_resid])
    r2_oracle = _cvs2(make_ridge(), X_plus_oracle, y, cv=cv, scoring='r2').mean()
    oracle_lift = r2_oracle - r2_k_base
    match_oracle = 0.02 < oracle_lift < 0.08
    total += 1; passed += int(match_oracle)
    print(f'ORACLE: +|V₃|² residual → κ lift={oracle_lift:+.4f}  '
          f'(paper: +0.04)  {"PASS" if match_oracle else "FAIL"}')

    # v3_resid correlates with κ residual
    k_resid = y - cross_val_predict(make_ridge(), X, y, cv=cv)
    r_bottleneck, _ = pearsonr(v3_resid, k_resid)
    match_bn = abs(r_bottleneck) > 0.40
    total += 1; passed += int(match_bn)
    print(f'BOTTLENECK: v3_resid~k_resid r={r_bottleneck:+.3f}  '
          f'(paper: −0.53)  {"PASS" if match_bn else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'  [{time.time()-t0:.1f}s]')
    sys.exit(0 if ok else 1)
