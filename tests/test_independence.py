"""§3.3 Weak coupling between stiffness and asymmetry — reproduces all numbers
and includes adversarial tests that attempt to falsify the independence claim.

Claims verified:
  - Pearson r(log fc_mean, log |V₃|²) = +0.09 (p = 0.15, not significant)
  - Partial r|mass = −0.05 (p = 0.46, n = 231)
  - Spearman ρ = +0.17 raw, +0.04 after mass control (mass-driven)
  - Detectable |r| > 0.13 at 95% confidence (exact t-test)
  - Multi-partial|mass,sv2,θD = −0.28 (exists but doesn't help κ prediction)
  - fc_mean removal changes residual-V₃ correlation by < 0.005
  - n = 231

Adversarial tests (attempt to falsify):
  - Spearman rank correlation (monotonic dependence)
  - Mutual information (nonlinear dependence)
  - Multi-variable partial (conditional coupling)
  - Bootstrap confidence interval
  - Practical impact: does fc_mean's conditional V₃ info help predict κ?

Usage: python test_independence.py

RAW RESULTS (2026-04-14, 231 materials)

PEARSON r=+0.095 p=0.151
PARTIAL|mass r=-0.049 p=0.46
SPEARMAN raw ρ=+0.168 p=0.010; partial|mass ρ=+0.037 p=0.579
DETECTABLE |r|=0.130 (exact t-test: 0.129)
MI=0.051 (not significant vs null 95th=0.058)
MULTI_PARTIAL|mass,sv2,θD r=-0.278 p<0.001
PRACTICAL fc_mean removal: residual-V₃ change = 0.004
BOOTSTRAP 95% CI: [-0.050, +0.231]
OVERALL: PASS (10/10)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata, t as t_dist
from numpy.linalg import lstsq
from _data import load_data, partial_corr, make_ridge, cv10, FEATURES_14


def _multi_partial(x, y, controls):
    Z = np.column_stack([controls, np.ones(len(x))])
    x_r = x - Z @ lstsq(Z, x, rcond=None)[0]
    y_r = y - Z @ lstsq(Z, y, rcond=None)[0]
    return pearsonr(x_r, y_r)


def main():
    m, mr = load_data()
    passed = 0
    total = 0

    print(f'RTA overlap: {len(mr)} materials')
    print()

    log_fc = np.log10(mr['fc_mean'].clip(lower=1e-3).values)
    log_V3sq = mr['log_V3sq'].values
    mass = mr['log_mass'].values
    log_sv2 = np.log10(mr['sv2_per_mode'].values)
    theta = mr['theta_D_m2_K'].values
    n = len(mr)

    # ── Dataset size ──────────────────────────────────────────────────
    match_n = n == 231
    total += 1; passed += int(match_n)
    print(f'N: {n}  (paper: 231)  {"PASS" if match_n else "FAIL"}')

    # ── Pearson raw correlation ───────────────────────────────────────
    r_raw, p_raw = pearsonr(log_fc, log_V3sq)
    match_raw = abs(r_raw - 0.09) < 0.02
    total += 1; passed += int(match_raw)
    print(f'PEARSON raw: r = {r_raw:+.3f}  p = {p_raw:.3f}  '
          f'(paper: +0.09, p = 0.15)  {"PASS" if match_raw else "FAIL"}')

    # ── Partial correlation|mass ──────────────────────────────────────
    r_part, p_part = partial_corr(log_fc, log_V3sq, mass)
    match_part = abs(r_part - (-0.05)) < 0.03 and p_part > 0.30
    total += 1; passed += int(match_part)
    print(f'PARTIAL|mass: r = {r_part:+.3f}  p = {p_part:.2f}  '
          f'(paper: −0.05, p = 0.46)  {"PASS" if match_part else "FAIL"}')

    # ── Spearman raw + partial|mass ──────────────────────────────────
    rho_raw, p_rho_raw = spearmanr(log_fc, log_V3sq)
    rank_fc = rankdata(log_fc)
    rank_V3 = rankdata(log_V3sq)
    rank_mass = rankdata(mass)
    rho_part, p_rho_part = _multi_partial(rank_fc, rank_V3, rank_mass.reshape(-1, 1))

    sp_raw_sig = p_rho_raw < 0.05
    sp_part_nonsig = p_rho_part > 0.30
    match_sp = sp_raw_sig and sp_part_nonsig
    total += 1; passed += int(match_sp)
    print(f'SPEARMAN raw: ρ = {rho_raw:+.3f}  p = {p_rho_raw:.3f}  (significant → mass-driven)')
    print(f'SPEARMAN|mass: ρ = {rho_part:+.3f}  p = {p_rho_part:.3f}  '
          f'(paper: not significant after mass control)  {"PASS" if match_sp else "FAIL"}')

    # ── Detectable bound (exact t-test) ──────────────────────────────
    tcrit = t_dist.ppf(0.975, df=n - 2)
    r_det_exact = np.sqrt(tcrit**2 / (tcrit**2 + (n - 2)))
    r_det_approx = 1.96 / np.sqrt(n - 3)
    match_det = abs(r_det_exact - 0.13) < 0.02
    total += 1; passed += int(match_det)
    print(f'DETECTABLE |r| (exact t): {r_det_exact:.3f}  '
          f'(approx: {r_det_approx:.3f})  (paper: 0.13)  {"PASS" if match_det else "FAIL"}')

    # ── Partial below detectable ─────────────────────────────────────
    below = abs(r_part) < r_det_exact
    total += 1; passed += int(below)
    print(f'|partial| < detectable: {abs(r_part):.3f} < {r_det_exact:.3f}  '
          f'{"PASS" if below else "FAIL"}')

    print()
    print('── Adversarial tests ──')
    print()

    # ── Mutual information ───────────────────────────────────────────
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(log_fc.reshape(-1, 1), log_V3sq,
                                random_state=42, n_neighbors=5)[0]
    mi_null = []
    for i in range(200):
        perm = np.random.RandomState(i).permutation(log_V3sq)
        mi_null.append(mutual_info_regression(
            log_fc.reshape(-1, 1), perm, random_state=42, n_neighbors=5)[0])
    mi_p95 = np.percentile(mi_null, 95)
    mi_nonsig = mi <= mi_p95
    total += 1; passed += int(mi_nonsig)
    print(f'MI: {mi:.4f}  (null 95th: {mi_p95:.4f})  '
          f'not significant → no nonlinear dependence  {"PASS" if mi_nonsig else "FAIL"}')

    # ── Multi-partial|mass,sv2,θD ────────────────────────────────────
    r_multi, p_multi = _multi_partial(log_fc, log_V3sq,
                                      np.column_stack([mass, log_sv2, theta]))
    match_multi = abs(r_multi - (-0.28)) < 0.05 and p_multi < 0.001
    total += 1; passed += int(match_multi)
    print(f'MULTI_PARTIAL|mass,sv2,θD: r = {r_multi:+.3f}  p = {p_multi:.1e}  '
          f'(paper: −0.28)  {"PASS" if match_multi else "FAIL"}')

    # ── Practical impact: fc_mean removal ────────────────────────────
    from sklearn.model_selection import cross_val_predict
    feats_no_fc = [f for f in FEATURES_14 if f != 'fc_mean']
    cv = cv10()
    pred_no_fc = cross_val_predict(make_ridge(), mr[feats_no_fc].values,
                                   mr['log_k'].values, cv=cv)
    pred_with_fc = cross_val_predict(make_ridge(), mr[FEATURES_14].values,
                                     mr['log_k'].values, cv=cv)
    r_resid_no, _ = pearsonr(mr['log_k'].values - pred_no_fc, log_V3sq)
    r_resid_with, _ = pearsonr(mr['log_k'].values - pred_with_fc, log_V3sq)
    change = abs(abs(r_resid_no) - abs(r_resid_with))
    match_prac = change < 0.01
    total += 1; passed += int(match_prac)
    print(f'PRACTICAL: resid~V₃ without fc_mean: r = {r_resid_no:+.3f}  '
          f'with: r = {r_resid_with:+.3f}  '
          f'change: {change:.4f}  (paper: <0.005)  {"PASS" if match_prac else "FAIL"}')

    # ── Bootstrap CI ─────────────────────────────────────────────────
    boot = []
    for i in range(2000):
        idx = np.random.RandomState(i).choice(n, n, replace=True)
        boot.append(pearsonr(log_fc[idx], log_V3sq[idx])[0])
    ci = np.percentile(boot, [2.5, 97.5])
    ci_includes_zero = ci[0] < 0 < ci[1]
    total += 1; passed += int(ci_includes_zero)
    print(f'BOOTSTRAP 95% CI: [{ci[0]:+.3f}, {ci[1]:+.3f}]  '
          f'includes 0? {ci_includes_zero}  '
          f'{"PASS" if ci_includes_zero else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'  [{time.time()-t0:.1f}s]')
    sys.exit(0 if ok else 1)
