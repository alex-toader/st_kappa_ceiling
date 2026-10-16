"""§3.2 Characterizing the residual — reproduces all numbers from the paper.

Claims verified:
  - 5-feature residual vs log(Γ): r = −0.54
  - 14-feature residual vs log(Γ): r = −0.28 (p < 10⁻⁴)
  - ANOVA across dominant-element groups: p = 0.04–0.07 over the fold assignments
  - Durbin-Watson: 1.97–2.04 ordered by predicted κ and v_LA; 1.77 ordered by mass (permutation p ≤ 0.002 on every assignment), 1.91 with one residual per composition (p = 0.11)
  - §4 factorization: log(κ) ≈ +1.11·log(sv2) − 0.80·log(Γ), R²_insample = 0.78, R²_CV = 0.74 ± 0.14
  - §3.5 oracle: the Γ/W scalar's residual improves κ by +0.05 R², 0.80 → 0.85

Usage: python test_residual.py

RAW RESULTS (749 materials / 229 RTA overlap; every cross-validated number the mean over ten 10-fold assignments, seeds 0–9)

RESID_5FEAT vs Γ: r = -0.542  (paper: -0.54)  PASS
RESID_14FEAT vs Γ: r = -0.278  largest p over the assignments 2.8e-05  (paper: -0.28, p < 10⁻⁴)  PASS
ANOVA (9 groups, largest O n=309): F=1.91  p 0.041 … 0.067 over the assignments  (paper: p = 0.04–0.07, nine groups)  PASS
DW by pred_κ    : 2.038
DW by log_mass  : 1.768
DW by log_v_LA  : 1.974
DURBIN_WATSON: by mass permutation p largest over the assignments 0.0017; one residual per composition (700) 1.906, p 0.106  (paper: 1.97–2.04; mass 1.77, p ≤ 0.002; per composition 1.91, p = 0.11)  PASS
FACTORIZATION: R²_insample = 0.7765  R²_CV = 0.7394 ± 0.1355  a = +1.111  b = -0.795  (paper: 0.78 / 0.74 ± 0.14, 1.11, -0.80)  PASS
ORACLE (Γ/W scalar): +|V₃|² residual → κ lift=+0.0529  (paper: 0.05, 0.80 → 0.85; base 0.7954, with 0.8483)  PASS
OVERALL: PASS  (6/6)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.linear_model import LinearRegression
from _data import (load_data, make_ridge, rcv_r2, rcv_mean, kfold, SEEDS, at_printed, composition, FEATURES_14, FEATURES_5)

# the numbers as the text prints them
TEXT = {'r5': '-0.54', 'r14': '-0.28', 'anova_lo': '0.04', 'anova_hi': '0.07', 'dw_min': '1.97', 'dw_max': '2.04', 'dw_mass': '1.77', 'dw_comp': '1.91', 'dw_comp_p': '0.11', 'fact_in': '0.78', 'fact_cv': '0.74', 'fact_sd': '0.14',
        'a': '1.11', 'b': '-0.80', 'oracle': '0.05', 'base': '0.80', 'with': '0.85'}


def nested_v3(mr, cv):
    """The |V₃|² oracle NESTED in one fold assignment: in each outer fold |V₃|² is fitted on the training rows only; a training row's
    residual is its inner out-of-fold one (the same assignment's KFold inside the training part), a test row's the outer fit's.
    Returns the fold-mean R² of κ for: the 14 features, + the |V₃|² residual."""
    from sklearn.metrics import r2_score
    y, X, yv = mr['log_k'].values, mr[FEATURES_14].values, mr['log_V3sq'].values
    sc = {'base': [], 'resid': []}
    for tr, te in cv.split(X):
        v_tr = cross_val_predict(make_ridge(), X[tr], yv[tr], cv=cv)
        v_te = make_ridge().fit(X[tr], yv[tr]).predict(X[te])
        for k, f_tr, f_te in (('base', X[tr], X[te]),
                              ('resid', np.column_stack([X[tr], yv[tr] - v_tr]), np.column_stack([X[te], yv[te] - v_te]))):
            sc[k].append(r2_score(y[te], make_ridge().fit(f_tr, y[tr]).predict(f_te)))
    return tuple(float(np.mean(sc[k])) for k in ('base', 'resid'))


def main():
    from _data import print_stack
    print_stack()
    m, mr = load_data()
    passed = 0
    total = 0
    print(f'RTA overlap: {len(mr)} materials')
    print()

    y = mr['log_k'].values
    gamma = mr['log_gamma'].values

    # ── Residual vs Γ (paper §3.2, second paragraph): the residual of each fold assignment, its r averaged over the assignments ──
    def resid_r(feats):
        def one(cv):
            r, p_ = pearsonr(gamma, y - cross_val_predict(make_ridge(), mr[feats].values, y, cv=cv))
            return r, p_
        vals = [one(kfold(s_)) for s_ in SEEDS]
        return float(np.mean([v[0] for v in vals])), float(max(v[1] for v in vals))
    r5, _ = resid_r(FEATURES_5)
    match5 = at_printed(r5, TEXT['r5'])
    total += 1; passed += match5
    print(f'RESID_5FEAT vs Γ: r = {r5:.3f}  (paper: {TEXT["r5"]})  {"PASS" if match5 else "FAIL"}')
    r14, p14 = resid_r(FEATURES_14)
    match14 = at_printed(r14, TEXT['r14']) and p14 < 1e-4
    total += 1; passed += match14
    print(f'RESID_14FEAT vs Γ: r = {r14:.3f}  largest p over the assignments {p14:.1e}  (paper: {TEXT["r14"]}, p < 10⁻⁴)  {"PASS" if match14 else "FAIL"}')
    print()

    # ── Residual uniformity (paper §3.2, last paragraph), on the full set ──
    from scipy.stats import f_oneway
    import re as _re

    def durbin_watson(resid):
        return np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)

    def _dom_el(formula):
        """The element with the largest stoichiometric count (parentheses expanded); a tie goes to the element named first in the formula."""
        c = composition(formula)
        order = list(dict.fromkeys(_re.findall(r'[A-Z][a-z]?', formula)))
        return max(order, key=lambda el: (c[el], -order.index(el)))
    classes = m['material'].apply(_dom_el)
    counts = classes.value_counts()
    major = counts[counts >= 15].index
    X_full, y_full = m[FEATURES_14].values, m['log_k'].values
    p_an, f_an, dws = [], [], {'pred_κ': [], 'log_mass': [], 'log_v_LA': []}
    for s_ in SEEDS:
        pred_full = cross_val_predict(make_ridge(), X_full, y_full, cv=kfold(s_))
        rf = y_full - pred_full
        f_, p_ = f_oneway(*[rf[(classes == cl).values] for cl in major])
        f_an.append(f_); p_an.append(p_)
        for oname, order in (('pred_κ', np.argsort(pred_full)), ('log_mass', np.argsort(m['log_mass'].values)), ('log_v_LA', np.argsort(m['log_v_LA'].values))):
            dws[oname].append(durbin_watson(rf[order]))
    match_a = at_printed(min(p_an), TEXT['anova_lo']) and at_printed(max(p_an), TEXT['anova_hi']) and len(major) == 9
    total += 1; passed += int(match_a)
    print(f'ANOVA ({len(major)} groups, largest {counts.index[0]} n={counts.max()}): F={np.mean(f_an):.2f}  p {min(p_an):.3f} … {max(p_an):.3f} over the assignments  '
          f'(paper: p = {TEXT["anova_lo"]}–{TEXT["anova_hi"]}, nine groups)  {"PASS" if match_a else "FAIL"}')
    dw = {k: float(np.mean(v)) for k, v in dws.items()}
    for oname, v in dw.items():
        print(f'DW by {oname:10s}: {v:.3f}')
    # the mass ordering: a permutation p per assignment (the largest reported), and one residual per composition (polymorphs share a mass and sit side by side)
    rng = np.random.default_rng(0)
    comp_key = m['material'].apply(lambda f_: tuple(sorted(composition(f_).items()))).values
    p_mass, dw_c, p_c = [], [], []
    for s_ in SEEDS:
        rf = y_full - cross_val_predict(make_ridge(), X_full, y_full, cv=kfold(s_))
        d_m = durbin_watson(rf[np.argsort(m['log_mass'].values)])
        p_mass.append((1 + sum(durbin_watson(rng.permutation(rf)) <= d_m for _ in range(4000))) / 4001)                # (k + 1) / (n + 1)
        g_ = pd.DataFrame({'c': comp_key, 'lm': m['log_mass'].values, 'r': rf}).groupby('c').agg(lm=('lm', 'first'), r=('r', 'mean')).sort_values('lm', kind='stable')
        rc = g_['r'].values
        dw_c.append(durbin_watson(rc))
        p_c.append((1 + sum(durbin_watson(rng.permutation(rc)) <= dw_c[-1] for _ in range(2000))) / 2001)
    others = [dw['pred_κ'], dw['log_v_LA']]
    dw_ok = (at_printed(min(others), TEXT['dw_min']) and at_printed(max(others), TEXT['dw_max']) and at_printed(dw['log_mass'], TEXT['dw_mass']) and max(p_mass) <= 0.002                                   # the text: p ≤ 0.002 on every assignment
             and at_printed(float(np.mean(dw_c)), TEXT['dw_comp']) and at_printed(float(np.mean(p_c)), TEXT['dw_comp_p']))
    total += 1; passed += dw_ok
    print(f'DURBIN_WATSON: by mass permutation p largest over the assignments {max(p_mass):.4f}; one residual per composition ({len(rc)}) {np.mean(dw_c):.3f}, p {np.mean(p_c):.3f}  '
          f'(paper: {TEXT["dw_min"]}–{TEXT["dw_max"]}; mass {TEXT["dw_mass"]}, p ≤ 0.002; per composition {TEXT["dw_comp"]}, p = {TEXT["dw_comp_p"]})  {"PASS" if dw_ok else "FAIL"}')

    # ── §4 factorization: κ ∝ sv2/Γ ──────────────────────────────────
    log_sv2 = np.log10(mr['sv2_per_mode'].values)
    X_fact = np.column_stack([log_sv2, gamma])
    reg = LinearRegression().fit(X_fact, y)
    r2_fact = reg.score(X_fact, y)
    a, b = reg.coef_
    r2_fact_cv, r2_fact_cv_sd, _ = rcv_r2(LinearRegression, X_fact, y)
    match_fact = (at_printed(r2_fact, TEXT['fact_in']) and at_printed(r2_fact_cv, TEXT['fact_cv']) and at_printed(r2_fact_cv_sd, TEXT['fact_sd']) and
                  at_printed(a, TEXT['a']) and at_printed(b, TEXT['b']))
    total += 1; passed += match_fact
    print(f'FACTORIZATION: R²_insample = {r2_fact:.4f}  R²_CV = {r2_fact_cv:.4f} ± {r2_fact_cv_sd:.4f}  a = {a:+.3f}  b = {b:+.3f}  '
          f'(paper: {TEXT["fact_in"]} / {TEXT["fact_cv"]} ± {TEXT["fact_sd"]}, {TEXT["a"]}, {TEXT["b"]})  {"PASS" if match_fact else "FAIL"}')

    # ── §3.5 oracle test: the Γ/W scalar's increment (nested) ──
    print()
    base, with_resid = rcv_mean(lambda cv: nested_v3(mr, cv))
    lift_resid = with_resid - base
    match_oracle = at_printed(lift_resid, TEXT['oracle']) and at_printed(base, TEXT['base']) and at_printed(with_resid, TEXT['with'])
    total += 1; passed += int(match_oracle)
    print(f'ORACLE (Γ/W scalar): +|V₃|² residual → κ lift={lift_resid:+.4f}  (paper: {TEXT["oracle"]}, {TEXT["base"]} → {TEXT["with"]}; base {base:.4f}, with {with_resid:.4f})  '
          f'{"PASS" if match_oracle else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'  [{time.time()-t0:.1f}s]')
    sys.exit(0 if ok else 1)
