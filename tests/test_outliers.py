"""§3.4 Material classes at the boundary — reproduces all numbers.

Claims verified:
  - Rigid oxides (TiO₂, ReO₃, ZrO₂, SnO₂): |V₃|² 1.36 std above mean, p = 0.006
  - Layered (GaSe×2, GaS, GaTe, WS₂, MoS₂, WSe₂, BiI): |V₃|² 1.60 std below mean, p < 0.001
  - Both survive mass control (p = 0.005, p < 0.001)

Usage: python test_outliers.py

RAW RESULTS (2026-04-14, 231 materials)

RIGID_OXIDE (n=4): z=+1.36  p=0.0060  mass_ctrl p=0.0042
LAYERED (n=8): z=-1.60  p<0.000001  mass_ctrl p=0.000001
OVERALL: PASS (6/6)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.stats import mannwhitneyu
from numpy.linalg import lstsq
from _data import load_data


RIGID_OXIDE_NAMES = ['TiO2', 'ReO3', 'ZrO2', 'SnO2']
LAYERED_NAMES = ['GaSe', 'GaS', 'GaTe', 'WS2', 'MoS2', 'WSe2', 'BiI']


def main():
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
    print(f'RTA overlap: {len(mr)} materials')
    print(f'Rigid oxides found: {n_rigid}  ({mr[is_rigid]["material"].tolist()})')
    print(f'Layered found: {n_layer}  ({mr[is_layer]["material"].tolist()})')
    print()

    # ── Rigid oxides: high |V₃|² ──────────────────────────────────────
    rigid_v3 = v3[is_rigid]
    other_v3 = v3[is_other]
    rigid_z = (rigid_v3.mean() - v3_mean) / v3_std
    _, p_rigid = mannwhitneyu(rigid_v3, other_v3, alternative='greater')

    match_z = abs(rigid_z - 1.36) < 0.15
    match_p = p_rigid < 0.05
    total += 1; passed += int(match_z)
    total += 1; passed += int(match_p)
    print(f'RIGID_OXIDE z-score: {rigid_z:+.2f}  (paper: +1.36)  '
          f'{"PASS" if match_z else "FAIL"}')
    print(f'RIGID_OXIDE p-value: {p_rigid:.4f}  (paper: 0.006)  '
          f'{"PASS" if match_p else "FAIL"}')

    # ── Layered: low |V₃|² ───────────────────────────────────────────
    layer_v3 = v3[is_layer]
    layer_z = (layer_v3.mean() - v3_mean) / v3_std
    _, p_layer = mannwhitneyu(layer_v3, other_v3, alternative='less')

    match_lz = abs(layer_z - (-1.60)) < 0.15
    match_lp = p_layer < 0.001
    total += 1; passed += int(match_lz)
    total += 1; passed += int(match_lp)
    print(f'LAYERED z-score: {layer_z:+.2f}  (paper: −1.60)  '
          f'{"PASS" if match_lz else "FAIL"}')
    print(f'LAYERED p-value: {p_layer:.6f}  (paper: < 0.001)  '
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

    match_rm = p_rigid_m < 0.05
    match_lm = p_layer_m < 0.001
    total += 1; passed += int(match_rm)
    total += 1; passed += int(match_lm)
    print(f'RIGID_OXIDE mass-controlled p: {p_rigid_m:.4f}  '
          f'(paper: 0.005)  {"PASS" if match_rm else "FAIL"}')
    print(f'LAYERED mass-controlled p: {p_layer_m:.6f}  '
          f'(paper: < 0.001)  {"PASS" if match_lm else "FAIL"}')

    print(f'\nOVERALL: {"PASS" if passed == total else "FAIL"}  ({passed}/{total})')
    return passed == total


if __name__ == '__main__':
    import time
    t0 = time.time()
    ok = main()
    print(f'  [{time.time()-t0:.1f}s]')
    sys.exit(0 if ok else 1)
