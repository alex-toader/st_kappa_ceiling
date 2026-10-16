"""§3.1 / §3.4 / §3.5 robustness checks — reproduces the numbers the text states.

Claims verified:
  - Grouped cross-validation (composition; chemical system; structural prototype = space group + anonymous formula) moves the pooled R² by at most 0.02 (Ridge, GB)
  - Learning curve (the design of test_ceiling.py's Ridge curve): GB still gains 0.03 between ≈ 400 and ≈ 670 training materials
  - Oracle with the coarse RTA κ: 14 features + log κ_RTA raises the subset R² from 0.80 to 0.91 (+0.11); log κ_RTA alone 0.90 (fold-mean R²)
  - R² as a range statistic (GB): linear-scale R² 0.77; within 1–10 W/mK R² 0.35 at RMSE 0.21 dex
  - Nearest-neighbour estimate across structural prototypes: the closest 5 / 10 / 20 % of the unique pairs differ by 0.17–0.20 dex (rms / √2)
  - Slack baseline with the acoustic Debye temperature (n exponent −5/3): R² 0.40 (Slack's exponents), 0.48 (one fitted scale), 0.59 (every exponent fitted)
  - Outlier classes: layered z −1.60 [−1.84, −1.41], rigid oxide +1.36 [+0.60, +2.22] (bootstrap); rule-based classes n 8 (p 0.009) and n 18 (p 0.0001); without the named members n 4 (p 0.19) and n 11 (p 0.08)
  - The RTA runs against the labels (§2.4): sd 0.19 dex, 0.13 without SrS₂ (run 477 W/mK, label 3.35 W/mK)
  - Label noise against Phonix (the 571 materials analysed here): sd 0.208 dex, robust 0.078; R² caps 0.91 / 0.96 / 0.99; the 0.96 cap's bootstrap 95 % interval 0.93–0.98 (68 % printed)
  - The dataset: 749 materials, the six crystal-system counts, κ 0.01–449 W/mK (4.7 orders), sd log κ 0.706 (population), 56 % in 1–10 W/mK
  - RMSE on the full set: GB 0.24, Ridge 0.26 dex
  - The 14-feature CV residual correlates with log |V₃|² at r = −0.27 (§3.2)
  - Ranking: Spearman ρ 0.92–0.93 over all materials, 0.72–0.74 within 1–10 W/mK (Ridge, GB); median relative error 30–34 % (GB 0.30, Ridge 0.34)
  - Partial r(fc_mean, |V₃|² | mass) = −0.05, 95 % CI −0.18 to +0.08 (Fisher z, n 229)
  - Label-noise second route: on the 571 materials in both sets, PhononDB's label error ≈ 0.093 dex (Ridge) / 0.050 (GB); cap 0.98
  - Imaginary modes (§2.1): 78 of the 749 DFT harmonic records, 46 at ≥ 100 mode points, ReO₃ among them; Ridge R² without the 78: 0.87
  - Tested descriptors (§4): Morse γ ρ +0.23 with the 14-feature residual, κ lift +0.008; element fractions → log |V₃|² R² 0.51, κ lift +0.001 (fitted inside each training fold)
  - The RTA table and the convergence artifacts (§2.1, §2.4): 247 rows, 238 below 10³ W/mK; the three artifacts' runs not finite, BaO₂ κ_zz 1.3 × 10⁴, Phonix 19× / 46× lower, Ridge R² 0.82 with them; six other diverged runs keep labels (four at 2–4 W/mK, LiC 13, BeSe 102)

Usage: python test_robustness.py

RAW RESULTS (749 materials / 229 RTA overlap; every cross-validated number the mean over ten 10-fold assignments, seeds 0–9)

GROUPED_CV Ridge: KFold 0.863  formula 0.864  chemical system 0.863  prototype 0.855  (text: moves ≤ 0.02)  PASS
GROUPED_CV GB   : KFold 0.883  formula 0.880  chemical system 0.873  prototype 0.870  (text: moves ≤ 0.02)  PASS
LEARNING_CURVE GB: gain 404→674 +0.0282 (range over the assignments +0.0181 … +0.0355)  (text: 0.03)  PASS
ORACLE_RTA: 14 0.795  14 + log κ_RTA 0.910 (+0.115)  log κ_RTA alone 0.896  (text: 0.80 → 0.91, alone 0.90)  PASS
RANGE: linear R² 0.765; band 1–10 W/mK n 418 R² 0.350 RMSE 0.213 dex  (text: 0.77; 0.35 at 0.21)  PASS
KNN_FLOOR: 592 unique cross-prototype pairs; closest 5 / 10 / 20 %: 0.167 / 0.186 / 0.201 dex  (text: 0.17–0.20)  PASS
SLACK: exponents fixed 0.402, one fitted scale 0.482, exponents fitted 0.593  (text: 0.40 / 0.48 / 0.59)  PASS
LAYERED z: -1.60, bootstrap 95 % [-1.84, -1.41] (n 8)  PASS
RIGID_OXIDE z: +1.36, bootstrap 95 % [+0.60, +2.22] (n 4)  PASS
STRUCTURAL_CLASSES: rigid-oxide-like n 8 p 0.0094; layered-like n 18 p 0.00012; without the named members n 4 p 0.190, n 11 p 0.079  (text: 8 at 0.009, 18 at 0.0001; 4 of 4 and 7 of 8 shared; 4 at 0.19, 11 at 0.08)  PASS
RTA_GAP: sd(log κ − log κ_RTA) 0.192 dex, without SrS2 0.127; SrS2 run 476.7 W/mK, label 3.35  (text: 0.19, 0.13; 477, 3.35)  PASS
PHONIX: 571 materials, sd 0.208 dex, robust 0.078; noise caps all-PhononDB / split / robust 0.913 / 0.956 / 0.988; split cap 68 % [0.945, 0.968], 95 % [0.930, 0.976]  (text: 571, 0.208, 0.078, 0.147 each, sd 0.706; 0.91 / 0.96 / 0.99; 0.93–0.98)  PASS
PHONIX_COVARIANCE: 571 materials; PhononDB label error ≈ 0.093 dex (Ridge) / 0.050 (GB); cap 0.983  (text: 571, 0.093 / 0.050, 0.98)  PASS
IMAGINARY: 78 with imaginary modes, 46 at ≥ 100; ReO3 [1864]; Ridge R² without them 0.8739  (text: 78, 46, ReO₃, 0.87)  PASS
RESID_V3: r(14-feature CV residual, log |V₃|²) = -0.269, largest p 6.8e-05  (text: −0.27, p < 10⁻⁴)  PASS
SPEARMAN: GB all 0.927 band 0.741, Ridge all 0.921 band 0.722; median relative error GB 0.301, Ridge 0.338  (text: 0.92–0.93 / 0.72–0.74; 30–34 %)  PASS
PARTIAL_CI: partial r -0.047, 95 % CI [-0.176, +0.084]  (text: −0.05, −0.18 to +0.08)  PASS
DATASET: 749 materials, crystal systems {'Orthorhombic': 208, 'Monoclinic': 174, 'Cubic': 136, 'Trigonal': 104, 'Tetragonal': 85, 'Hexagonal': 42}; κ 0.0100–449.42 W/mK, span 4.653 dex; sd log κ 0.7060; 1–10 W/mK 55.8 %  (text: 766 → 749 by 2 + 3 + 2 + 10; 208 / 174 / 136 / 104 / 85 / 42; 0.01–449, 4.7 orders; 0.706; 56 %; Hg / Yb / Pa in 20; interaction range 1.0 in 77 %)  PASS
RMSE: GB 0.2411, Ridge 0.2612 dex  (text: 0.24, 0.26)  PASS
TESTED_DESCRIPTORS: Morse γ (k-avg) ρ with the 14-feature residual +0.230, κ lift +0.0081; element fractions → log |V₃|² R² 0.507, κ lift +0.0006  (text: +0.23, +0.008; 0.51, +0.001)  PASS
ARTIFACTS: 247 rows, 238 below 1e3; 6x6x6 κ inf, inf, inf; BaO2 κ_zz 13478, Phonix 2.2e+05; label / Phonix 19.1, 46.5; the three largest residuals True; Ridge R² with them 0.815; the other 6 diverging runs keep labels [np.float64(2.09), np.float64(101.54), np.float64(3.69), np.float64(3.59), np.float64(2.53), np.float64(13.45)]  (text: 247, 238; not finite, 1.3e4, 19x / 46x, 0.82, six: four at 2-4, LiC 13, BeSe 102)  PASS
OVERALL: PASS  (21/21)
"""
import sys
import os
import re
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict, cross_val_score, GroupKFold, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from _data import load_data, make_ridge, make_gb, kfold, SEEDS, rcv_r2, rcv_mean, FEATURES_14, DATA_DIR, partial_corr, at_printed, composition, _compute_delta_en
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

N_MAIN = 749
N_BAND = 418
LAYERED_NAMES = ['GaSe', 'GaS', 'GaTe', 'WS2', 'MoS2', 'WSe2', 'BiI']
RIGID_NAMES = ['TiO2', 'ReO3', 'ZrO2', 'SnO2']


def pooled_r2(y, p):
    return 1 - np.mean((y - p) ** 2) / np.var(y)


def rcv_pooled(make_model, X, y):
    """The pooled out-of-fold R², averaged over the ten fold assignments."""
    return float(np.mean([pooled_r2(y, cross_val_predict(make_model(), X, y, cv=kfold(s))) for s in SEEDS]))


def anonymous(formula):
    n = sorted(composition(formula).values())
    g = np.gcd.reduce(n)
    return tuple(v // g for v in n)


def main():
    from _data import print_stack
    print_stack()
    m, mr = load_data()
    m = m.reset_index(drop=True)
    sg = pd.read_csv(os.path.join(DATA_DIR, 'spacegroup_phonopy_v1.csv'))
    assert not sg.mp_id.duplicated().any(), 'spacegroup_phonopy_v1.csv has duplicate mp_id'
    m = m.merge(sg[['mp_id', 'spacegroup', 'volume', 'nsites']], on='mp_id', how='left')
    assert len(m) == N_MAIN, f'{len(m)} rows after the merge, expected {N_MAIN}'
    assert m.spacegroup.notna().all() and m.volume.notna().all() and m.nsites.notna().all(), 'a material lacks its space group, volume or site count'
    X, y = m[FEATURES_14].values, m['log_k'].values
    proto = np.array([f'{s}|{anonymous(f)}' for s, f in zip(m.spacegroup, m.material)])
    chem = np.array(['-'.join(sorted(set(re.findall(r'[A-Z][a-z]?', f)))) for f in m.material])
    passed = total = 0
    print(f'{len(m)} materials, {len(mr)} RTA overlap, {len(set(proto))} structural prototypes')
    print()

    # ── Grouped cross-validation (polymorph / prototype leakage) ───────
    for name, mk in [('Ridge', make_ridge), ('GB', make_gb)]:
        base = rcv_pooled(mk, X, y)
        g_form = pooled_r2(y, cross_val_predict(mk(), X, y, cv=GroupKFold(10), groups=m.material.values))
        g_chem = pooled_r2(y, cross_val_predict(mk(), X, y, cv=GroupKFold(10), groups=chem))
        g_proto = pooled_r2(y, cross_val_predict(mk(), X, y, cv=GroupKFold(10), groups=proto))
        want = ('0.86',) if name == 'Ridge' else ('0.87', '0.88')                       # the text: Ridge 0.86, gradient boosting 0.87–0.88
        ok = max(base - g_form, base - g_chem, base - g_proto) <= 0.0205 and all(any(at_printed(g_, w) for w in want) for g_ in (g_form, g_chem, g_proto))
        total += 1; passed += ok
        print(f'GROUPED_CV {name:5s}: KFold {base:.3f}  formula {g_form:.3f}  chemical system {g_chem:.3f}  prototype {g_proto:.3f}  (text: moves ≤ 0.02)  {"PASS" if ok else "FAIL"}')

    # ── Gradient boosting's learning curve (the design of test_ceiling.py's Ridge curve) ──
    # each held-out fold is kept; the model trains on a random 60 % of its training fold (≈ 400 materials) and on all of it; shuffle=True, so the subset is random
    gains = []
    for s_ in SEEDS:
        sizes, _, te = learning_curve(make_gb(), X, y, cv=kfold(s_), train_sizes=[.6, 1.0], scoring='r2', shuffle=True, random_state=s_)
        gains.append(te.mean(1)[1] - te.mean(1)[0])
    g_lc = float(np.mean(gains))
    ok = at_printed(g_lc, '0.03') and tuple(sizes) == (404, 674)                                  # the text's ≈ 400 and ≈ 670 training materials
    total += 1; passed += ok
    print(f'LEARNING_CURVE GB: gain {sizes[0]}→{sizes[1]} {g_lc:+.4f} (range over the assignments {min(gains):+.4f} … {max(gains):+.4f})  (text: 0.03)  {"PASS" if ok else "FAIL"}')

    # ── Oracle with the coarse RTA κ (229-material subset) ─────────────
    ys = mr['log_k'].values
    r14 = rcv_r2(make_ridge, mr[FEATURES_14].values, ys)[0]
    r14k = rcv_r2(make_ridge, mr[FEATURES_14 + ['log_krta']].values, ys)[0]
    rk = rcv_r2(make_ridge, mr[['log_krta']].values, ys)[0]
    ok = at_printed(r14, '0.80') and at_printed(r14k, '0.91') and at_printed(rk, '0.90') and at_printed(r14k - r14, '0.11')
    total += 1; passed += ok
    print(f'ORACLE_RTA: 14 {r14:.3f}  14 + log κ_RTA {r14k:.3f} ({r14k - r14:+.3f})  log κ_RTA alone {rk:.3f}  (text: 0.80 → 0.91, alone 0.90)  {"PASS" if ok else "FAIL"}')

    # ── R² as a range statistic ────────────────────────────────────────
    P_GB = {s_: cross_val_predict(make_gb(), X, y, cv=kfold(s_)) for s_ in SEEDS}               # gradient boosting's out-of-fold predictions per assignment (reused below)
    P_RIDGE = {s_: cross_val_predict(make_ridge(), X, y, cv=kfold(s_)) for s_ in SEEDS}
    k = 10 ** y
    band = (k >= 1) & (k <= 10)
    lin = float(np.mean([1 - np.mean((k - 10 ** q) ** 2) / np.var(k) for q in P_GB.values()]))
    r2b = float(np.mean([pooled_r2(y[band], q[band]) for q in P_GB.values()]))
    rmseb = float(np.mean([np.sqrt(np.mean((y[band] - q[band]) ** 2)) for q in P_GB.values()]))
    ok = at_printed(lin, '0.77') and at_printed(r2b, '0.35') and at_printed(rmseb, '0.21') and band.sum() == N_BAND
    total += 1; passed += ok
    print(f'RANGE: linear R² {lin:.3f}; band 1–10 W/mK n {band.sum()} R² {r2b:.3f} RMSE {rmseb:.3f} dex  (text: 0.77; 0.35 at 0.21)  {"PASS" if ok else "FAIL"}')

    # ── Nearest-neighbour floor across structural prototypes ───────────
    Z = StandardScaler().fit_transform(X)
    d, idx = NearestNeighbors(n_neighbors=60).fit(Z).kneighbors(Z)
    pairs = {}
    for i in range(len(y)):
        for c in range(1, 60):
            j = idx[i, c]
            if proto[j] != proto[i]:
                pairs[tuple(sorted((i, j)))] = d[i, c]                                        # each unordered pair once
                break
    pi = np.array([a for a, _ in pairs]); pj = np.array([b for _, b in pairs]); pd_ = np.array(list(pairs.values()))
    floors = {}
    for q in (0.05, 0.10, 0.20):
        close = pd_ <= np.quantile(pd_, q)
        floors[q] = float(np.sqrt(np.mean((y[pi[close]] - y[pj[close]]) ** 2) / 2))
    fv = list(floors.values())
    ok = at_printed(min(fv), '0.17') and at_printed(max(fv), '0.20') and len(pairs) == 592
    total += 1; passed += ok
    print(f'KNN_FLOOR: {len(pairs)} unique cross-prototype pairs; closest 5 / 10 / 20 %: ' + ' / '.join(f'{f:.3f}' for f in floors.values())
          + f' dex  (text: 0.17–0.20)  {"PASS" if ok else "FAIL"}')

    # ── Slack-type baseline ────────────────────────────────────────────
    delta = np.log10((m.volume / m.nsites) ** (1 / 3))
    n = np.log10(m.n_atoms_primitive)
    xs = np.log10(m.avg_mass_AMU) + 3 * np.log10(m.theta_D_m2_K) + delta - (5 / 3) * n                 # θ_a = θ_D n^(−1/3): θ_a³ n^(−2/3) = θ_D³ n^(−5/3)
    xs = xs.values
    free = np.column_stack([np.log10(m.avg_mass_AMU), np.log10(m.theta_D_m2_K), delta, n])

    def slack_exact(cv):                                                  # Slack's exponents and a fitted constant only
        p_ = np.empty(len(y))
        for tr, te in cv.split(xs):
            p_[te] = xs[te] + np.mean(y[tr] - xs[tr])
        return pooled_r2(y, p_)
    se_ = rcv_mean(slack_exact)
    sf = rcv_pooled(LinearRegression, xs[:, None], y)                      # Slack's exponents with one fitted overall scale
    sv = rcv_pooled(LinearRegression, free, y)                            # every exponent fitted
    ok = at_printed(se_, '0.40') and at_printed(sf, '0.48') and at_printed(sv, '0.59')
    total += 1; passed += ok
    print(f'SLACK: exponents fixed {se_:.3f}, one fitted scale {sf:.3f}, exponents fitted {sv:.3f}  (text: 0.40 / 0.48 / 0.59)  {"PASS" if ok else "FAIL"}')

    # ── Outlier classes: bootstrap intervals, and the rule-based classes ─
    v3 = mr['log_V3sq'].values
    rng = np.random.default_rng(0)
    for tag, names, n_want, z_want, side in (('LAYERED', LAYERED_NAMES, 8, -1.60, 'hi'), ('RIGID_OXIDE', RIGID_NAMES, 4, 1.36, 'lo')):
        cls = mr['material'].isin(names).values
        assert cls.sum() == n_want, f'{tag}: {cls.sum()} members, expected {n_want}'
        z0 = (v3[cls].mean() - v3.mean()) / v3.std()
        zs = []
        for _ in range(2000):
            b = rng.integers(0, len(v3), len(v3))
            if cls[b].sum() >= 2:
                zs.append((v3[b][cls[b]].mean() - v3[b].mean()) / v3[b].std())
        lo, hi = np.quantile(zs, [0.025, 0.975])
        ci_want = {'LAYERED': ('-1.84', '-1.41'), 'RIGID_OXIDE': ('0.60', '2.22')}[tag]
        ok = at_printed(z0, str(z_want)) and at_printed(lo, ci_want[0]) and at_printed(hi, ci_want[1]) and cls.sum() == n_want
        total += 1; passed += ok
        print(f'{tag} z: {z0:+.2f}, bootstrap 95 % [{lo:+.2f}, {hi:+.2f}] (n {cls.sum()})  {"PASS" if ok else "FAIL"}')
    q75 = m['fc_mean'].quantile(0.75)                                                          # the rule of st_3's harmonic–anharmonic test C3
    rigid_like = ((mr.fc_mean > q75) & mr.crystal_system.isin(['Cubic', 'Tetragonal', 'Orthorhombic']) & (mr.n_atoms_primitive >= 3) & (mr.n_atoms_primitive <= 12)).values
    layer_like = (mr.crystal_system.isin(['Hexagonal', 'Trigonal']) & (mr.interaction_range < 0.95)).values
    other = ~rigid_like & ~layer_like
    p_r = mannwhitneyu(v3[rigid_like], v3[other], alternative='greater')[1]
    p_l = mannwhitneyu(v3[layer_like], v3[other], alternative='less')[1]
    named_r, named_l = mr['material'].isin(RIGID_NAMES).values, mr['material'].isin(LAYERED_NAMES).values      # the rule classes without the named lists' members
    r_only, l_only = rigid_like & ~named_r, layer_like & ~named_l
    p_r0 = mannwhitneyu(v3[r_only], v3[other], alternative='greater')[1]
    p_l0 = mannwhitneyu(v3[l_only], v3[other], alternative='less')[1]
    ok = (rigid_like.sum() == 8 and layer_like.sum() == 18 and at_printed(p_r, '0.009') and at_printed(p_l, '0.0001')
          and (rigid_like & named_r).sum() == 4 and named_r.sum() == 4 and (layer_like & named_l).sum() == 7 and named_l.sum() == 8
          and r_only.sum() == 4 and l_only.sum() == 11 and at_printed(p_r0, '0.19') and at_printed(p_l0, '0.08'))
    total += 1; passed += ok
    print(f'STRUCTURAL_CLASSES: rigid-oxide-like n {rigid_like.sum()} p {p_r:.4f}; layered-like n {layer_like.sum()} p {p_l:.5f}; without the named members n {r_only.sum()} p {p_r0:.3f}, '
          f'n {l_only.sum()} p {p_l0:.3f}  (text: 8 at 0.009, 18 at 0.0001; 4 of 4 and 7 of 8 shared; 4 at 0.19, 11 at 0.08)  {"PASS" if ok else "FAIL"}')

    # ── The RTA runs against the labels (§2.4) ──────────────────────────
    gap = mr['log_k'] - mr['log_krta']
    noise, noise_wo = float(gap.std()), float(gap[mr['material'] != 'SrS2'].std())
    srs2 = mr[mr['material'] == 'SrS2']
    assert len(srs2) == 1, 'SrS2 is not once in the RTA subset'
    k_run, k_lab = float(10 ** srs2['log_krta'].iloc[0]), float(10 ** srs2['log_k'].iloc[0])
    ok = at_printed(noise, '0.19') and at_printed(noise_wo, '0.13') and at_printed(k_run, '477') and at_printed(k_lab, '3.35')
    total += 1; passed += ok
    print(f'RTA_GAP: sd(log κ − log κ_RTA) {noise:.3f} dex, without SrS2 {noise_wo:.3f}; SrS2 run {k_run:.1f} W/mK, label {k_lab:.2f}  (text: 0.19, 0.13; 477, 3.35)  {"PASS" if ok else "FAIL"}')

    # ── Label noise against Phonix (independent database) ──────────────
    ph = pd.read_csv(os.path.join(DATA_DIR, 'phonix_kp_overlap_v1.csv'))
    kb = pd.read_csv(os.path.join(DATA_DIR, 'kappa_rta_phonondb.csv'))
    g = ph.groupby('mp_id')['kp_W_mK'].apply(lambda v: np.log10(v).mean()).rename('lkp').reset_index().merge(kb[['mp_id', 'k_rta_avg']], on='mp_id')
    g = g[g.mp_id.astype(int).isin(m.mp_id.astype(int))]                                    # the materials analysed here
    dlt = np.log10(g.k_rta_avg) - g.lkp
    sd_, rob = float(dlt.std()), float(1.4826 * np.median(np.abs(dlt - np.median(dlt))))
    caps = [1 - (s_ / y.std()) ** 2 for s_ in (sd_, sd_ / np.sqrt(2), rob)]
    rng_p, dv = np.random.default_rng(0), dlt.values                                       # the split cap's bootstrap: the 571 differences and the 749 labels resampled
    split_b = np.array([1 - (dv[rng_p.integers(0, len(dv), len(dv))].std(ddof=1) / np.sqrt(2) / y[rng_p.integers(0, len(y), len(y))].std()) ** 2 for _ in range(4000)])
    q68, q95 = np.quantile(split_b, [0.16, 0.84]), np.quantile(split_b, [0.025, 0.975])
    plateau_gap = (caps[1] - 0.89, caps[1] - 0.86)                                         # the plateau's printed ends (test_ceiling) below the central cap: the text's 0.07–0.10
    ok = at_printed(plateau_gap[0], '0.07') and at_printed(plateau_gap[1], '0.10') and len(g) == 571 and at_printed(sd_, '0.208') and at_printed(rob, '0.078') and at_printed(sd_ / np.sqrt(2), '0.147') and at_printed(y.std(), '0.706') \
         and at_printed(caps[0], '0.91') and at_printed(caps[1], '0.96') and at_printed(caps[2], '0.99') and at_printed(q95[0], '0.93') and at_printed(q95[1], '0.98')
    total += 1; passed += ok
    print(f'PHONIX: {len(g)} materials, sd {sd_:.3f} dex, robust {rob:.3f}; noise caps all-PhononDB / split / robust {caps[0]:.3f} / {caps[1]:.3f} / {caps[2]:.3f}; '
          f'split cap 68 % [{q68[0]:.3f}, {q68[1]:.3f}], 95 % [{q95[0]:.3f}, {q95[1]:.3f}]  (text: 571, 0.208, 0.078, 0.147 each, sd 0.706; 0.91 / 0.96 / 0.99; 0.93–0.98)  {"PASS" if ok else "FAIL"}')

    # ── PhononDB's own label error, from the residual's covariance with the PhononDB − Phonix difference (a second route to the noise bound) ──
    gm = m[['mp_id', 'log_k']].reset_index(drop=True)
    bound = {}
    for name_, P_ in (('Ridge', P_RIDGE), ('GB', P_GB)):
        res_ = np.mean([y - q for q in P_.values()], axis=0)                                  # the covariance is linear in the residual: its mean over the assignments is this one's
        mm2 = gm.assign(res=res_).merge(g[['mp_id', 'lkp']], on='mp_id', validate='one_to_one')
        r_, d_ = mm2.res.values, (mm2.log_k - mm2.lkp).values
        c_ = float(np.cov(r_, d_)[0, 1])
        bound[name_] = (len(mm2), np.sqrt(c_), 1 - c_ / float(np.var(y, ddof=1)))
    ok = bound['Ridge'][0] == 571 and at_printed(bound['Ridge'][1], '0.093') and at_printed(bound['GB'][1], '0.050') and at_printed(bound['Ridge'][2], '0.98')
    total += 1; passed += ok
    print(f'PHONIX_COVARIANCE: {bound["Ridge"][0]} materials; PhononDB label error ≈ {bound["Ridge"][1]:.3f} dex (Ridge) / {bound["GB"][1]:.3f} (GB); cap {bound["Ridge"][2]:.3f}  '
          f'(text: 571, 0.093 / 0.050, 0.98)  {"PASS" if ok else "FAIL"}')

    # ── Imaginary modes in the DFT harmonic records (§2.1, §3.4) ───────
    mesh_ = pd.read_csv(os.path.join(DATA_DIR, 'features_mesh_v1.csv'))[['mp_id', 'n_imag_modes']]
    mi = m[['mp_id', 'material']].merge(mesh_, on='mp_id', how='left', validate='one_to_one')
    assert mi.n_imag_modes.notna().all()
    imag = (mi.n_imag_modes > 0).values
    r2_noimag = rcv_r2(make_ridge, X[~imag], y[~imag])[0]
    ok = imag.sum() == 78 and int((mi.n_imag_modes >= 100).sum()) == 46 and bool((mi[mi.material == 'ReO3'].n_imag_modes > 0).all()) and at_printed(r2_noimag, '0.87')
    total += 1; passed += ok
    print(f'IMAGINARY: {imag.sum()} with imaginary modes, {int((mi.n_imag_modes >= 100).sum())} at ≥ 100; ReO3 {mi[mi.material == "ReO3"].n_imag_modes.tolist()}; Ridge R² without them {r2_noimag:.4f}  '
          f'(text: 78, 46, ReO₃, 0.87)  {"PASS" if ok else "FAIL"}')

    # ── Residual against log |V₃|² (§3.2) ──────────────────────────────
    ys = mr['log_k'].values
    rv_ = [pearsonr(ys - cross_val_predict(make_ridge(), mr[FEATURES_14].values, ys, cv=kfold(s_)), mr['log_V3sq'].values) for s_ in SEEDS]
    r_rv, p_rv = float(np.mean([v[0] for v in rv_])), float(max(v[1] for v in rv_))
    ok = at_printed(r_rv, '-0.27') and p_rv < 1e-4
    total += 1; passed += ok
    print(f'RESID_V3: r(14-feature CV residual, log |V₃|²) = {r_rv:+.3f}, largest p {p_rv:.1e}  (text: −0.27, p < 10⁻⁴)  {"PASS" if ok else "FAIL"}')

    # ── Ranking (§4, practical implications) ───────────────────────────
    rho = {n: (float(np.mean([spearmanr(y, q)[0] for q in P_.values()])), float(np.mean([spearmanr(y[band], q[band])[0] for q in P_.values()])))
           for n, P_ in (('GB', P_GB), ('Ridge', P_RIDGE))}
    ok = all((at_printed(a, '0.92') or at_printed(a, '0.93')) and (at_printed(b, '0.72') or at_printed(b, '0.73') or at_printed(b, '0.74')) for a, b in rho.values())
    mre = {n: float(np.mean([np.median(np.abs(10 ** q - k) / k) for q in P_.values()])) for n, P_ in (('GB', P_GB), ('Ridge', P_RIDGE))}   # the median relative error, κ in W/mK
    ok = ok and at_printed(mre['GB'], '0.30') and at_printed(mre['Ridge'], '0.34')                          # the text: 30–34 %
    total += 1; passed += ok
    print(f'SPEARMAN: ' + ', '.join(f'{n} all {a:.3f} band {b:.3f}' for n, (a, b) in rho.items()) + '; median relative error ' + ', '.join(f'{n} {v:.3f}' for n, v in mre.items())
          + f'  (text: 0.92–0.93 / 0.72–0.74; 30–34 %)  {"PASS" if ok else "FAIL"}')

    # ── Partial r(fc_mean, |V₃|² | mass) with its interval (§3.3) ───────
    rp = float(partial_corr(np.log10(mr['fc_mean'].values), mr['log_V3sq'].values, mr['log_mass'].values)[0])   # fc_mean > 1e-3 asserted in test_independence
    z, se = np.arctanh(rp), 1 / np.sqrt(len(mr) - 3 - 1)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    ok = at_printed(rp, '-0.05') and at_printed(lo, '-0.18') and at_printed(hi, '0.08')
    total += 1; passed += ok
    print(f'PARTIAL_CI: partial r {rp:+.3f}, 95 % CI [{lo:+.3f}, {hi:+.3f}]  (text: −0.05, −0.18 to +0.08)  {"PASS" if ok else "FAIL"}')

    # ── The dataset as §2.1 / §3.1 / the abstract describe it ──────────
    k = m['k_rta_avg'].values
    lk = np.log10(k)
    cs = m['crystal_system'].value_counts().to_dict()
    band_share = float(np.mean((k >= 1) & (k <= 10)))
    ok = (len(m) == N_MAIN and cs == {'Orthorhombic': 208, 'Monoclinic': 174, 'Cubic': 136, 'Trigonal': 104, 'Tetragonal': 85, 'Hexagonal': 42}
          and at_printed(lk.max() - lk.min(), '4.7') and at_printed(k.min(), '0.01') and at_printed(k.max(), '449')
          and at_printed(float(lk.std()), '0.706') and at_printed(100 * band_share, '56'))
    kb = pd.read_csv(os.path.join(DATA_DIR, 'kappa_rta_phonondb.csv'))
    gone = kb[~kb.mp_id.astype(int).isin(m.mp_id.astype(int))]
    named = {'NaLa2RuO6', 'Rb2H6Pt', 'BaO2', 'Pr2C(NO)2', 'Na2PdC2', 'K2O', 'Al(HO)3'}
    rest = gone[~gone.material.isin(named)]
    en_out = rest.material.apply(lambda f_: bool(np.isnan(_compute_delta_en(f_)))).all()
    en_kinds = sorted(rest.material.apply(lambda f_: 'Hg' if 'Hg' in composition(f_) else 'Yb' if 'Yb' in composition(f_) else f_))   # B, two Si, three Hg, four Yb
    en_ok = en_kinds.count('Hg') == 3 and en_kinds.count('Yb') == 4 and en_kinds.count('Si') == 2 and en_kinds.count('B') == 1
    hyp = m.material.apply(lambda f_: bool({'Hg', 'Yb', 'Pa'} & set(composition(f_)))).sum()
    single = m.material.apply(lambda f_: len(composition(f_)) == 1).sum()                      # the rule: no single-element solid among the 749
    ok = ok and single == 0 and len(kb) == 766 and len(gone) == 17 and named <= set(gone.material) and len(rest) == 10 and en_out and en_ok and hyp == 20 \
        and at_printed(100 * float((m['interaction_range'] == 1.0).mean()), '77')
    total += 1; passed += ok
    print(f'DATASET: {len(m)} materials, crystal systems {cs}; κ {k.min():.4f}–{k.max():.2f} W/mK, span {lk.max() - lk.min():.3f} dex; sd log κ {lk.std():.4f}; '
          f'1–10 W/mK {100 * band_share:.1f} %  (text: 766 → 749 by 2 + 3 + 2 + 10; 208 / 174 / 136 / 104 / 85 / 42; 0.01–449, 4.7 orders; 0.706; 56 %; Hg / Yb / Pa in 20; interaction range 1.0 in 77 %)  {"PASS" if ok else "FAIL"}')

    # ── RMSE on the full set (§3.1 (iii)) ───────────────────────────────
    rm = {n: float(np.mean([np.sqrt(np.mean((q - y) ** 2)) for q in P_.values()])) for n, P_ in (('GB', P_GB), ('Ridge', P_RIDGE))}
    ok = at_printed(rm['GB'], '0.24') and at_printed(rm['Ridge'], '0.26')
    total += 1; passed += ok
    print(f'RMSE: GB {rm["GB"]:.4f}, Ridge {rm["Ridge"]:.4f} dex  (text: 0.24, 0.26)  {"PASS" if ok else "FAIL"}')

    # ── Tested descriptors that do not bridge the gap (§4) ──────────────
    def _comp(f):
        """Stoichiometric element fractions (parentheses expanded by _data.composition)."""
        c = composition(f)
        tot = sum(c.values())
        return {k_: v_ / tot for k_, v_ in c.items()}
    morse = pd.read_csv(os.path.join(DATA_DIR, 'features_morse_analytical_v0.csv'))
    morse['mp_id'] = morse['mp_id'].astype(int)
    mm_ = m.merge(morse, on='mp_id', how='left', validate='one_to_one')
    assert mm_['gamma_morse_kavg'].notna().all() and len(mm_) == len(m)
    rho_m = float(np.mean([spearmanr(mm_['gamma_morse_kavg'], y - q)[0] for q in P_RIDGE.values()]))   # residual = log κ − prediction (Fig. 2's sign): a Grüneisen effect would give ρ < 0
    print(f'  (residual convention log κ − predicted; Morse γ against log κ itself ρ {spearmanr(mm_["gamma_morse_kavg"], y)[0]:+.3f})')
    lift_m = rcv_r2(make_ridge, np.column_stack([X, mm_['gamma_morse_kavg'].values]), y)[0] - rcv_r2(make_ridge, X, y)[0]
    E_ = pd.DataFrame([_comp(f_) for f_ in mr.material]).fillna(0.0).values
    yv, Xr, yr = mr['log_V3sq'].values, mr[FEATURES_14].values, mr['log_k'].values
    r2_el = rcv_r2(make_ridge, E_, yv)[0]

    def el_lift(cv):                                                      # nested: the element-fraction prediction of |V₃|² fitted on each outer training fold only
        from sklearn.metrics import r2_score
        b_, w_ = [], []
        for tr, te in cv.split(Xr):
            v_tr = cross_val_predict(make_ridge(), E_[tr], yv[tr], cv=cv)
            v_te = make_ridge().fit(E_[tr], yv[tr]).predict(E_[te])
            b_.append(r2_score(yr[te], make_ridge().fit(Xr[tr], yr[tr]).predict(Xr[te])))
            w_.append(r2_score(yr[te], make_ridge().fit(np.column_stack([Xr[tr], v_tr]), yr[tr]).predict(np.column_stack([Xr[te], v_te]))))
        return float(np.mean(w_) - np.mean(b_))
    lift_el = rcv_mean(el_lift)
    ok = at_printed(rho_m, '0.23') and at_printed(lift_m, '0.008') and at_printed(r2_el, '0.51') and at_printed(lift_el, '0.001')
    total += 1; passed += ok
    print(f'TESTED_DESCRIPTORS: Morse γ (k-avg) ρ with the 14-feature residual {rho_m:+.3f}, κ lift {lift_m:+.4f}; element fractions → log |V₃|² R² {r2_el:.3f}, '
          f'κ lift {lift_el:+.4f}  (text: +0.23, +0.008; 0.51, +0.001)  {"PASS" if ok else "FAIL"}')

    # ── The three κ convergence artifacts: the evidence the text gives, and the sensitivity ──
    kb = pd.read_csv(os.path.join(DATA_DIR, 'kappa_rta_phonondb.csv')).set_index('mp_id')
    r6 = pd.read_csv(os.path.join(DATA_DIR, 'features_scattering_rta_v1.csv')).set_index('mp_id')
    px = pd.read_csv(os.path.join(DATA_DIR, 'phonix_kp_overlap_v1.csv')).set_index('mp_id')
    import _data as _dd
    art = sorted(_dd.KAPPA_ARTIFACTS)
    div = r6[r6.kappa_rta >= 1000]
    ratio = {i: kb.loc[i, 'k_rta_avg'] / px.loc[i, 'kp_W_mK'] for i in (4823, 9497)}
    kept = [i for i in div.index if i not in _dd.KAPPA_ARTIFACTS]
    kept_px = [kb.loc[i, 'k_rta_avg'] / px.loc[i, 'kp_W_mK'] for i in kept if i in px.index]
    labs = sorted(float(kb.loc[i, 'k_rta_avg']) for i in kept)
    _saved = set(_dd.KAPPA_ARTIFACTS)
    try:
        _dd.KAPPA_ARTIFACTS.clear()
        m_all, _ = _dd.load_data()
    finally:
        _dd.KAPPA_ARTIFACTS.update(_saved)
    Xa, ya = m_all[FEATURES_14].values, m_all['log_k'].values
    r2_all = rcv_r2(make_ridge, Xa, ya)[0]
    res_all = np.mean([np.abs(ya - cross_val_predict(make_ridge(), Xa, ya, cv=kfold(s_))) for s_ in SEEDS], axis=0)
    top3 = set(m_all.mp_id.values[np.argsort(-res_all)[:3]])
    n_rows, n_conv = len(r6), int((r6.kappa_rta < 1000).sum())
    ok = (len(m_all) == N_MAIN + 3 and n_rows == 247 and n_conv == 238 and all(not np.isfinite(r6.loc[i, 'kappa_rta']) for i in art) and at_printed(kb.loc[1105, 'k_rta_zz'] / 1e4, '1.3') and px.loc[1105, 'kp_W_mK'] > 1e5
          and at_printed(ratio[4823], '19') and at_printed(ratio[9497], '46') and top3 == set(art) and at_printed(r2_all, '0.82')
          and len(kept) == 6 and all(not np.isfinite(r6.loc[i, 'kappa_rta']) or r6.loc[i, 'kappa_rta'] >= 1e23 for i in kept)
          and sum(2 <= x <= 4 for x in labs) == 4 and at_printed(kb.loc[1378, 'k_rta_avg'], '13') and at_printed(kb.loc[1541, 'k_rta_avg'], '102') and {1378, 1541} <= set(kept) and all(0.5 <= q <= 2 for q in kept_px))
    total += 1; passed += ok
    print(f'ARTIFACTS: {n_rows} rows, {n_conv} below 1e3; 6x6x6 κ {", ".join(f"{r6.loc[i, 'kappa_rta']:.1e}" for i in art)}; BaO2 κ_zz {kb.loc[1105, 'k_rta_zz']:.0f}, Phonix {px.loc[1105, 'kp_W_mK']:.1e}; '
          f'label / Phonix {ratio[4823]:.1f}, {ratio[9497]:.1f}; the three largest residuals {top3 == set(art)}; Ridge R² with them {r2_all:.3f}; '
          f'the other {len(kept)} diverging runs keep labels {[round(kb.loc[i, "k_rta_avg"], 2) for i in kept]}  (text: 247, 238; not finite, 1.3e4, 19x / 46x, 0.82, six: four at 2-4, LiC 13, BeSe 102)  {"PASS" if ok else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'({time.time() - t0:.0f} s)')
    sys.exit(0 if ok else 1)
