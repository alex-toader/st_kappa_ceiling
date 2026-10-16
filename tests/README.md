# Tests — reproducibility for the paper

55 checks across 5 test files, on the 749-material dataset (229 for the RTA overlap). Every cross-validated number is the mean over ten fold assignments of 10-fold cross-validation (KFold, shuffled, seeds 0–9), except the grouped cross-validation (one GroupKFold assignment per grouping); the nearest-neighbour estimate uses no cross-validation; the ± of an R² is the standard deviation over all 100 folds.

## Run all

```bash
for f in tests/test_*.py; do python3 "$f" || exit 1; done
```

## What each file verifies

### `test_ceiling.py` — §3.1 (12 checks)

- **R² plateau**: Ridge 0.86±0.03, GB 0.88±0.03, MLP 0.89±0.03 on 749 materials; the ten assignments' Ridge means within 0.01
- **Ablation**: velocities 0.61 → +Debye 0.64 → +spectral 0.79 → +structural 0.84 → +W 0.86
- **W marginal lift**: +0.02 controlling for the other 13 features (the same comparison as the ablation's last step)
- **Robustness**: 31 features (+17 engineered) → Ridge 0.89, MLP 0.91, MLP gain ≈ 0.02
- **Learning curve (Ridge)**: trained on 60 % of each training fold (≈400) and on all of it, tested on the same held-out fold: gain <0.01

### `test_residual.py` — §3.2 + §3.5 + §4 (6 checks)

- **Residual vs log(Γ)**: r=−0.54 (5-feat baseline), r=−0.28 (14-feat, p<10⁻⁴), the residual cross-validated on the 229
- **ANOVA across 9 dominant-element groups**: p=0.04–0.07 over the fold assignments
- **Durbin-Watson**: 1.97–2.04 ordered by predicted κ and v_LA; 1.77 ordered by mass (permutation p ≤ 0.002 on every assignment), 1.91 with one residual per composition (p = 0.11)
- **Factorization κ ≈ A·sv2·Γ⁻¹**: R²=0.78 in-sample, 0.74 ± 0.14 CV, a=+1.11, b=−0.80
- **Oracle test (Γ/W scalar)**: |V₃|² residual (fitted inside each training fold) → κ lift = +0.05, 0.80 → 0.85 (the RTA-κ oracle is in test_robustness.py)

### `test_independence.py` — §3.3 (8 checks)

- **n=229 RTA overlap**
- **Pearson r(log fc_mean, log |V₃|²)**: +0.10 (p=0.15, not significant)
- **Partial|mass**: r=−0.05 (p=0.48)
- **Spearman**: ρ=+0.17 raw (p=0.01), +0.04 after mass control (p=0.54)
- **Detectable |r|**: 0.13 at 95% (exact t-test), ≈ 0.18 for 80 % power; the partial r below it (a check that follows from the two numbers)
- **Multi-partial|mass,sv2,θD**: r=−0.28 (p<10⁻⁴, conditional)
- **Practical impact**: removing fc_mean changes residual-V₃ correlation by 0.007

### `test_outliers.py` — §3.4 (8 checks)

- **Rigid oxides** (TiO₂ and SnO₂ rutile, ReO₃ Pm-3m, ZrO₂ P4₂/nmc; n=4): |V₃|² z=+1.36, Mann–Whitney p=0.006 against the other 217
- **Layered** (GaSe×2, GaS, GaTe, WS₂, MoS₂, WSe₂, BiI; n=8): |V₃|² z=−1.60, p<0.001
- **Mass-controlled**: rigid p=0.005, layered p<0.001
- **Residual sign** (14-feature Ridge cross-validated on the 229): TiO₂, ReO₃, ZrO₂ overpredicted by 0.7–1.3 dex, SnO₂ ≈ 0; six layered underpredicted by 0.13–0.46 dex, WS₂ ≈ 0, BiI overpredicted by 0.19
- **Degenerate-band treatment** (data/degenerate_weights_layered_v1.csv): averaging the tetrahedron weights over degenerate bands moves the eight layered materials' Γ_transport by at most 0.002 dex

### `test_robustness.py` — §2.1–§3.5, §4 (21 checks)

- **Dataset**: 766 → 749 (2 without a harmonic record, 3 convergence artifacts, 2 structure mismatches, 10 without an electronegativity contrast); crystal systems 208 / 174 / 136 / 104 / 85 / 42; κ 0.01–449 W/mK (4.7 orders); sd log κ 0.706; 56 % in 1–10 W/mK; the 10 without a contrast are B, two Si, three Hg and four Yb compounds; Hg / Yb / Pa in 20 retained materials; interaction range 1.0 in 77 %
- **RMSE**: GB 0.24, Ridge 0.26 dex (full set)
- **Tested descriptors (§4)**: Morse γ (k-avg) ρ +0.23 with the 14-feature residual, κ lift +0.008; element fractions → log |V₃|² R² 0.51, κ lift +0.001 (fitted inside each training fold)
- **Grouped CV**: by composition, chemical system and structural prototype (space group + anonymous formula), Ridge and GB; R² moves ≤ 0.02
- **Learning curve (GB)**: the design of the Ridge curve; GB still gains 0.03 between ≈400 and ≈670 training materials
- **Oracle with the coarse RTA κ**: 0.80 → 0.91 on the 229 subset; log κ_RTA alone 0.90
- **Error by range (GB)**: linear-scale R² 0.77; 1–10 W/mK R² 0.35 at RMSE 0.21 dex
- **Nearest-neighbour estimate**: 592 unique cross-prototype pairs; the closest 5–20 % differ by 0.17–0.20 dex (no cross-validation)
- **Slack baseline** (acoustic Debye temperature, n^(−5/3)): R² 0.40 with Slack's exponents, 0.48 with one fitted scale, 0.59 with every exponent fitted
- **Outlier classes**: layered z −1.60 [−1.84, −1.41], rigid oxide +1.36 [+0.60, +2.22]; rule-based classes n 8 (p 0.009), n 18 (p 0.0001); without the named members n 4 (p 0.19), n 11 (p 0.08)
- **RTA runs against the labels**: sd 0.19 dex, 0.13 without SrS₂ (run 477 W/mK, label 3.35)
- **Imaginary modes**: 78 of the 749 DFT harmonic records (46 at ≥ 100 mode points, ReO₃ among them); Ridge R² 0.87 without them
- **Phonix label noise** (data/phonix_kp_overlap_v1.csv, the 571 materials analysed here): sd 0.208 dex, robust 0.078; R² caps 0.91 / 0.96 / 0.99; the 0.96 cap's bootstrap 95 % interval 0.93–0.98
- **Phonix covariance** (571 materials): PhononDB's label error ≈ 0.093 dex (Ridge) / 0.050 (GB); cap 0.98
- **RTA table and convergence artifacts**: 247 rows, 238 below 10³ W/mK; the three artifacts' runs not finite, BaO₂ κ_zz 1.3 × 10⁴, Phonix 19× / 46× lower, Ridge R² 0.82 with them; six other diverged runs keep labels (four at 2–4 W/mK, LiC 13, BeSe 102)
- **Residual vs |V₃|²**: r = −0.27; **ranking**: Spearman 0.92–0.93 overall, 0.72–0.74 within 1–10 W/mK, median relative error 30 % (GB) to 34 % (Ridge); **partial r**: −0.05, 95 % CI −0.18 to +0.08

## Configuration

All tests use shared data loading and cross-validation from `_data.py`:
- 749 materials (after the κ-convergence-artifact, structure-mismatch and electronegativity-contrast exclusions)
- 229-material RTA overlap
- 14 standard features: sound velocities (3), Debye temperature, phase space W, spectral shape (5), structural (4)
- Ridge(α=1), GB(500 trees, depth 4), MLP(256-128-64, 1000 iter), model random state 42
- KFold(10, shuffle=True, random_state=s) for s in 0–9; numbers are means over the ten assignments
