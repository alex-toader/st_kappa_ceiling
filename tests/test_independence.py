"""§3.3 Bond stiffness and the coupling proxy — reproduces all numbers of the section.

Claims verified:
  - Pearson r(log fc_mean, log |V₃|²) = +0.10 (p = 0.15, not significant)
  - Partial r|mass = −0.05 (p = 0.48, n = 229)
  - Spearman ρ = +0.17 raw (p = 0.01), +0.04 after mass control (p = 0.54)
  - Detectable |r| > 0.13 at 95% confidence (exact t-test), ≈ 0.18 for 80 % power
  - Multi-partial|mass,sv2,θD = −0.28
  - fc_mean removal changes residual-V₃ correlation by 0.007
  - n = 229

Usage: python test_independence.py

RAW RESULTS (749 materials / 229 RTA overlap; every cross-validated number the mean over ten 10-fold assignments, seeds 0–9)

N: 229  (paper: 229)  PASS
PEARSON raw: r = +0.096  p = 0.147  (paper: +0.10, p = 0.15)  PASS
PARTIAL|mass: r = -0.047  p = 0.48  (paper: −0.05, p = 0.48)  PASS
SPEARMAN raw: ρ = +0.169  p = 0.010
SPEARMAN|mass: ρ = +0.041  p = 0.535  (paper: ρ +0.17, p = 0.01 before; +0.04, p = 0.54 after)  PASS
DETECTABLE |r| (exact t): 0.130  (approx: 0.130); 80 % power 0.184  (paper: 0.13, ≈ 0.18)  PASS
|partial| < detectable: 0.047 < 0.130  PASS
MULTI_PARTIAL|mass,sv2,θD: r = -0.277  p = 2.4e-05  (paper: −0.28)  PASS
PRACTICAL: resid~V₃ without fc_mean: r = -0.275  with: r = -0.269  change: 0.0068  (paper: 0.007)  PASS
OVERALL: PASS  (8/8)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import pearsonr, spearmanr, rankdata, t as t_dist
from numpy.linalg import lstsq
from _data import load_data, partial_corr, make_ridge, rcv_mean, FEATURES_14, at_printed


def main():
    from _data import print_stack
    print_stack()
    m, mr = load_data()
    passed = 0
    total = 0

    print(f'RTA overlap: {len(mr)} materials')
    print()

    assert (mr['fc_mean'] > 1e-3).all(), 'a material of the subset has fc_mean ≤ 1e-3: the log would be clipped'
    log_fc = np.log10(mr['fc_mean'].values)
    log_V3sq = mr['log_V3sq'].values
    mass = mr['log_mass'].values
    log_sv2 = np.log10(mr['sv2_per_mode'].values)
    theta = mr['theta_D_m2_K'].values
    n = len(mr)

    # ── Dataset size ──────────────────────────────────────────────────
    match_n = n == 229
    total += 1; passed += int(match_n)
    print(f'N: {n}  (paper: 229)  {"PASS" if match_n else "FAIL"}')

    # ── Pearson raw correlation ───────────────────────────────────────
    r_raw, p_raw = pearsonr(log_fc, log_V3sq)
    match_raw = at_printed(r_raw, '0.10') and at_printed(p_raw, '0.15')
    total += 1; passed += int(match_raw)
    print(f'PEARSON raw: r = {r_raw:+.3f}  p = {p_raw:.3f}  '
          f'(paper: +0.10, p = 0.15)  {"PASS" if match_raw else "FAIL"}')

    # ── Partial correlation|mass ──────────────────────────────────────
    r_part, p_part = partial_corr(log_fc, log_V3sq, mass)
    match_part = at_printed(r_part, '-0.05') and at_printed(p_part, '0.48')
    total += 1; passed += int(match_part)
    print(f'PARTIAL|mass: r = {r_part:+.3f}  p = {p_part:.2f}  '
          f'(paper: −0.05, p = 0.48)  {"PASS" if match_part else "FAIL"}')

    # ── Spearman raw + partial|mass ──────────────────────────────────
    rho_raw, p_rho_raw = spearmanr(log_fc, log_V3sq)
    rank_fc = rankdata(log_fc)
    rank_V3 = rankdata(log_V3sq)
    rank_mass = rankdata(mass)
    rho_part, p_rho_part = partial_corr(rank_fc, rank_V3, rank_mass)

    match_sp = at_printed(rho_raw, '0.17') and at_printed(p_rho_raw, '0.01') and at_printed(rho_part, '0.04') and at_printed(p_rho_part, '0.54')
    total += 1; passed += int(match_sp)
    print(f'SPEARMAN raw: ρ = {rho_raw:+.3f}  p = {p_rho_raw:.3f}')
    print(f'SPEARMAN|mass: ρ = {rho_part:+.3f}  p = {p_rho_part:.3f}  '
          f'(paper: ρ +0.17, p = 0.01 before; +0.04, p = 0.54 after)  {"PASS" if match_sp else "FAIL"}')

    # ── Detectable bound (exact t-test) ──────────────────────────────
    tcrit = t_dist.ppf(0.975, df=n - 2)
    r_det_exact = np.sqrt(tcrit**2 / (tcrit**2 + (n - 2)))
    r_det_approx = 1.96 / np.sqrt(n - 3)
    r_det80 = np.tanh((1.959964 + 0.841621) / np.sqrt(n - 3))                    # Fisher z: the |r| detected with 80 % power at two-sided α 0.05
    match_det = at_printed(r_det_exact, '0.13') and at_printed(r_det80, '0.18')
    total += 1; passed += int(match_det)
    print(f'DETECTABLE |r| (exact t): {r_det_exact:.3f}  '
          f'(approx: {r_det_approx:.3f}); 80 % power {r_det80:.3f}  (paper: 0.13, ≈ 0.18)  {"PASS" if match_det else "FAIL"}')

    # ── Partial below detectable ─────────────────────────────────────
    below = abs(r_part) < r_det_exact
    total += 1; passed += int(below)
    print(f'|partial| < detectable: {abs(r_part):.3f} < {r_det_exact:.3f}  '
          f'{"PASS" if below else "FAIL"}')

    print()
    print('── Adversarial tests ──')
    print()

    # ── Multi-partial|mass,sv2,θD ────────────────────────────────────
    r_multi, p_multi = partial_corr(log_fc, log_V3sq, np.column_stack([mass, log_sv2, theta]))
    match_multi = at_printed(r_multi, '-0.28') and p_multi < 1e-4
    total += 1; passed += int(match_multi)
    print(f'MULTI_PARTIAL|mass,sv2,θD: r = {r_multi:+.3f}  p = {p_multi:.1e}  '
          f'(paper: −0.28)  {"PASS" if match_multi else "FAIL"}')

    # ── Practical impact: fc_mean removal ────────────────────────────
    from sklearn.model_selection import cross_val_predict
    feats_no_fc = [f for f in FEATURES_14 if f != 'fc_mean']
    yk = mr['log_k'].values

    def both(cv):                                                         # the residual's r with |V₃|², without and with fc_mean, on one fold assignment
        return (pearsonr(yk - cross_val_predict(make_ridge(), mr[feats_no_fc].values, yk, cv=cv), log_V3sq)[0],
                pearsonr(yk - cross_val_predict(make_ridge(), mr[FEATURES_14].values, yk, cv=cv), log_V3sq)[0])
    r_resid_no, r_resid_with = rcv_mean(both)
    change = abs(abs(r_resid_no) - abs(r_resid_with))
    match_prac = at_printed(change, '0.007')
    total += 1; passed += int(match_prac)
    print(f'PRACTICAL: resid~V₃ without fc_mean: r = {r_resid_no:+.3f}  '
          f'with: r = {r_resid_with:+.3f}  '
          f'change: {change:.4f}  (paper: 0.007)  {"PASS" if match_prac else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'  [{time.time()-t0:.1f}s]')
    sys.exit(0 if ok else 1)
