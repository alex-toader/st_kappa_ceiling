"""§3.4 Material classes at the boundary — reproduces all numbers.

Claims verified:
  - Rigid oxides (TiO₂ and SnO₂ rutile P4₂/mnm, ReO₃ Pm-3m, ZrO₂ P4₂/nmc): |V₃|² 1.36 std above mean, p = 0.006 against the other 217
  - Layered (GaSe P-6m2 and P6₃/mmc, GaS, GaTe, WS₂, MoS₂, WSe₂, BiI): |V₃|² 1.60 std below mean, p < 0.001
  - z = (class mean − subset mean) / subset std, std over all 229 with ddof 0, class members included
  - Both survive mass control (p = 0.005, p < 0.001)
  - The 14-feature residual (Ridge cross-validated on the 229, averaged over the ten assignments): three rigid oxides overpredicted by 0.7–1.3 dex, SnO₂ ≈ 0; six layered underpredicted by 0.13–0.46 dex, WS₂ ≈ 0, BiI overpredicted by 0.19
  - Averaging the tetrahedron weights over degenerate bands (phono3py average_degenerate_weights) changes the eight layered materials' transport-weighted Γ by at most 0.002 dex

Usage: python test_outliers.py

RAW RESULTS (749 materials / 229 RTA overlap; every cross-validated number the mean over ten 10-fold assignments, seeds 0–9)

Members: BiI mp-568388 C2/m, GaS mp-2507 P6_3/mmc, GaSe mp-1572 P-6m2, GaSe mp-1943 P6_3/mmc, GaTe mp-10009 P6_3/mmc, MoS2 mp-1434 R3m, ReO3 mp-190 Pm-3m, SnO2 mp-856 P4_2/mnm, TiO2 mp-2657 P4_2/mnm, WS2 mp-224 P6_3/mmc, WSe2 mp-1821 P6_3/mmc, ZrO2 mp-2574 P4_2/nmc
RIGID_OXIDE z-score: +1.36  (paper: +1.36)  PASS
RIGID_OXIDE p-value: 0.0060 against the other 217  (paper: 0.006, 217)  PASS
LAYERED z-score: -1.60  (paper: −1.60)  PASS
LAYERED p-value: 4.2e-09  (paper: < 0.001)  PASS
RIGID_OXIDE mass-controlled p: 0.0046  (paper: 0.005)  PASS
LAYERED mass-controlled p: 8.4e-07  (paper: < 0.001)  PASS
RESIDUAL_SIGN: rigid ReO3 -0.911, SnO2 +0.025, TiO2 -1.319, ZrO2 -0.675; layered BiI -0.192, GaS +0.339, GaSe +0.355, GaSe +0.457, GaTe +0.171, MoS2 +0.196, WS2 +0.012, WSe2 +0.128  (paper: three overpredicted by 0.7–1.3, SnO2 ≈ 0; six underpredicted by 0.13–0.46, WS2 ≈ 0, BiI overpredicted by 0.19)  PASS
DEGENERATE_WEIGHTS: max |Δ log10 Γ_transport| over the 8 layered = 0.0017 dex (MoS2 0.0009); the layered offset in log |V₃|² -0.902 dex  (paper: at most 0.002)  PASS
OVERALL: PASS  (8/8)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from numpy.linalg import lstsq
from sklearn.model_selection import cross_val_predict
from _data import load_data, at_printed, DATA_DIR, SEEDS, kfold, make_ridge, FEATURES_14


RIGID_OXIDE_NAMES = ['TiO2', 'ReO3', 'ZrO2', 'SnO2']
LAYERED_NAMES = ['GaSe', 'GaS', 'GaTe', 'WS2', 'MoS2', 'WSe2', 'BiI']


def main():
    from _data import print_stack
    print_stack()
    m, mr = load_data()
    passed = 0
    total = 0

    v3 = mr['log_V3sq'].values
    v3_mean = v3.mean()
    v3_std = v3.std()
    mass = mr['log_mass'].values

    is_rigid = mr['material'].isin(RIGID_OXIDE_NAMES)
    is_layer = mr['material'].isin(LAYERED_NAMES)
    is_other = ~is_rigid & ~is_layer

    n_rigid = is_rigid.sum()
    n_layer = is_layer.sum()
    assert n_rigid == 4 and n_layer == 8 and int((mr.loc[is_layer, 'material'] == 'GaSe').sum()) == 2, f'class sizes {n_rigid} / {n_layer}, expected 4 / 8 with two GaSe'
    import pandas as pd
    from _data import DATA_DIR
    sg = pd.read_csv(os.path.join(DATA_DIR, 'spacegroup_phonopy_v1.csv'))[['mp_id', 'sg_symbol']]
    members = mr[is_rigid | is_layer][['mp_id', 'material']].merge(sg, on='mp_id', validate='one_to_one')
    want_sg = {('TiO2', 'P4_2/mnm'), ('SnO2', 'P4_2/mnm'), ('ReO3', 'Pm-3m'), ('ZrO2', 'P4_2/nmc'), ('GaSe', 'P-6m2'), ('GaSe', 'P6_3/mmc')}
    assert want_sg <= set(zip(members.material, members.sg_symbol)), 'a named member is not the polymorph the text names'
    print(f'RTA overlap: {len(mr)} materials')
    print('Members: ' + ', '.join(f'{r.material} mp-{r.mp_id} {r.sg_symbol}' for r in members.itertuples()))
    print()

    # ── Rigid oxides: high |V₃|² ──────────────────────────────────────
    rigid_v3 = v3[is_rigid]
    other_v3 = v3[is_other]
    rigid_z = (rigid_v3.mean() - v3_mean) / v3_std
    _, p_rigid = mannwhitneyu(rigid_v3, other_v3, alternative='greater')

    match_z = at_printed(rigid_z, '1.36')
    match_p = at_printed(p_rigid, '0.006') and int(is_other.sum()) == 217
    total += 1; passed += int(match_z)
    total += 1; passed += int(match_p)
    print(f'RIGID_OXIDE z-score: {rigid_z:+.2f}  (paper: +1.36)  '
          f'{"PASS" if match_z else "FAIL"}')
    print(f'RIGID_OXIDE p-value: {p_rigid:.4f} against the other {int(is_other.sum())}  (paper: 0.006, 217)  '
          f'{"PASS" if match_p else "FAIL"}')

    # ── Layered: low |V₃|² ───────────────────────────────────────────
    layer_v3 = v3[is_layer]
    layer_z = (layer_v3.mean() - v3_mean) / v3_std
    _, p_layer = mannwhitneyu(layer_v3, other_v3, alternative='less')

    match_lz = at_printed(layer_z, '-1.60')
    match_lp = p_layer < 0.001
    total += 1; passed += int(match_lz)
    total += 1; passed += int(match_lp)
    print(f'LAYERED z-score: {layer_z:+.2f}  (paper: −1.60)  '
          f'{"PASS" if match_lz else "FAIL"}')
    print(f'LAYERED p-value: {p_layer:.1e}  (paper: < 0.001)  '
          f'{"PASS" if match_lp else "FAIL"}')
    print()

    # ── Mass control ─────────────────────────────────────────────────
    z_arr = np.column_stack([mass, np.ones(len(mass))])
    v3_resid = v3 - z_arr @ lstsq(z_arr, v3, rcond=None)[0]

    rigid_rm = v3_resid[is_rigid]
    layer_rm = v3_resid[is_layer]
    other_rm = v3_resid[is_other]

    _, p_rigid_m = mannwhitneyu(rigid_rm, other_rm, alternative='greater')
    _, p_layer_m = mannwhitneyu(layer_rm, other_rm, alternative='less')

    match_rm = at_printed(p_rigid_m, '0.005')
    match_lm = p_layer_m < 0.001
    total += 1; passed += int(match_rm)
    total += 1; passed += int(match_lm)
    print(f'RIGID_OXIDE mass-controlled p: {p_rigid_m:.4f}  '
          f'(paper: 0.005)  {"PASS" if match_rm else "FAIL"}')
    print(f'LAYERED mass-controlled p: {p_layer_m:.1e}  '
          f'(paper: < 0.001)  {"PASS" if match_lm else "FAIL"}')

    # ── Sign of the harmonic residual per class member (§3.4) ───────────
    ys = mr['log_k'].values
    res = np.mean([ys - cross_val_predict(make_ridge(), mr[FEATURES_14].values, ys, cv=kfold(sd)) for sd in SEEDS], axis=0)   # log κ − predicted
    res_r, res_l = pd.Series(res[is_rigid.values], mr.loc[is_rigid, 'material'].values), pd.Series(res[is_layer.values], mr.loc[is_layer, 'material'].values)
    over_r = res_r.drop('SnO2')
    under_l = res_l.drop(['WS2', 'BiI'])
    match_sign = (bool((over_r < 0).all()) and at_printed(-over_r.max(), '0.7') and at_printed(-over_r.min(), '1.3') and at_printed(res_r['SnO2'], '0.0')
                  and bool((under_l > 0).all()) and len(under_l) == 6 and at_printed(under_l.min(), '0.13') and at_printed(under_l.max(), '0.46')
                  and at_printed(res_l['WS2'], '0.0') and at_printed(-res_l['BiI'], '0.19'))
    total += 1; passed += int(match_sign)
    print(f'RESIDUAL_SIGN: rigid {", ".join(f"{k} {v:+.3f}" for k, v in res_r.items())}; layered {", ".join(f"{k} {v:+.3f}" for k, v in res_l.items())}  '
          f'(paper: three overpredicted by 0.7–1.3, SnO2 ≈ 0; six underpredicted by 0.13–0.46, WS2 ≈ 0, BiI overpredicted by 0.19)  {"PASS" if match_sign else "FAIL"}')

    # ── Degenerate-band sensitivity of the layered Γ (§3.4, last sentence) ─
    dw = pd.read_csv(os.path.join(DATA_DIR, 'degenerate_weights_layered_v1.csv'))
    lay_ids = set(mr.loc[is_layer, 'mp_id'])
    if set(dw['mp_id']) != lay_ids:
        raise SystemExit(f'degenerate_weights_layered_v1.csv covers {sorted(dw.mp_id)}, the layered set is {sorted(lay_ids)}')
    rta = pd.read_csv(os.path.join(DATA_DIR, 'features_scattering_rta_v1.csv')).set_index('mp_id')
    if not np.array_equal(dw['gamma_transport_default'].values, rta.loc[dw['mp_id'], 'gamma_transport'].values):
        raise SystemExit('the default-weights Γ in degenerate_weights_layered_v1.csv differs from features_scattering_rta_v1.csv')
    dlog = np.log10(dw['gamma_transport_degenerate_averaged'].values / dw['gamma_transport_default'].values)
    offset = v3[is_layer].mean() - v3_mean
    match_dw = np.abs(dlog).max() <= 0.002 and at_printed(np.abs(dlog).max(), '0.002')   # 'at most 0.002'
    total += 1; passed += int(match_dw)
    print(f'DEGENERATE_WEIGHTS: max |Δ log10 Γ_transport| over the {len(dw)} layered = {np.abs(dlog).max():.4f} dex '
          f'(MoS2 {abs(dlog[dw.material.values == "MoS2"][0]):.4f}); the layered offset in log |V₃|² {offset:+.3f} dex  '
          f'(paper: at most 0.002)  {"PASS" if match_dw else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'  [{time.time()-t0:.1f}s]')
    sys.exit(0 if ok else 1)
