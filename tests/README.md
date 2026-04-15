# Tests — reproducibility for the paper

37 checks across 4 test files. All pass with fixed random seed (42) and 10-fold CV on the 751-material dataset (231 for RTA overlap).

## Run all

```bash
for f in tests/test_*.py; do python3 "$f"; done
```

## What each file verifies

### `test_ceiling.py` — §3.1 (12 checks)

- **R² ceiling**: Ridge 0.86±0.04, GB 0.88±0.02, MLP 0.89±0.03 on 751 materials
- **Ablation**: velocities 0.61 → +Debye 0.63 → +spectral 0.79 → +structural 0.83 → +W 0.86
- **W marginal lift**: +0.02 controlling for the other 13 features
- **Robustness**: 33 features (+19 engineered) → Ridge 0.89, MLP gain <0.02
- **Learning curve**: saturates by N≈400 (R² gain <0.01 from 400 to 751)

### `test_residual.py` — §3.2 + §3.5 (12 checks)

- **BTE/RTA unpredictable**: Ridge R²=−0.39, GB=−2.12, MLP=−1.48 (all negative)
- **Residual vs log(Γ)**: r=−0.54 (5-feat baseline), r=−0.29 (14-feat, p<10⁻⁴)
- **W partial|mass vs residual**: r=−0.01 (negligible)
- **ANOVA across 10 dominant-element groups**: p=0.07
- **Durbin-Watson autocorrelation**: 1.78–2.16 (no structure)
- **Factorization κ ≈ A·sv2·Γ⁻¹**: R²=0.78 in-sample, 0.73 CV, a=+1.11, b=−0.80
- **FC2 → |V₃|² at R²=0.70**, but κ lift = 0.000 (redundancy)
- **Oracle test**: true |V₃|² residual → κ lift = +0.04 (ceiling bound)
- **Bottleneck**: v3_residual ~ κ_residual at r=−0.53

### `test_independence.py` — §3.3 (10 checks)

- **n=231 RTA overlap**
- **Pearson r(log fc_mean, log |V₃|²)**: +0.09 (p=0.15, not significant)
- **Partial|mass**: r=−0.05 (p=0.46)
- **Spearman raw**: ρ=+0.17 (p=0.01, mass-driven)
- **Spearman|mass**: ρ=+0.04 (p=0.58)
- **Detectable |r|**: 0.13 at 95% (exact t-test)
- **Mutual information**: not significant vs permutation null (no nonlinear dependence)
- **Multi-partial|mass,sv2,θD**: r=−0.28 (p<10⁻⁴, conditional)
- **Practical impact**: removing fc_mean changes residual-V₃ correlation by <0.005
- **Bootstrap 95% CI**: [−0.05, +0.23] (includes 0)

### `test_outliers.py` — §3.4 (6 checks)

- **Rigid oxides** (TiO₂, ReO₃, ZrO₂, SnO₂; n=4): |V₃|² z=+1.36, Mann-Whitney p=0.006
- **Layered** (GaSe×2, GaS, GaTe, WS₂, MoS₂, WSe₂, BiI; n=8): |V₃|² z=−1.60, p<0.001
- **Mass-controlled**: rigid p=0.005, layered p<0.001

## Configuration

All tests use shared data loading from `_data.py`:
- 751 materials (after BTE-artifact and electronegativity-contrast exclusions)
- 231-material RTA overlap
- 14 standard features: sound velocities (3), Debye temperature, phase space W, spectral shape (5), structural (4)
- Ridge(α=1), GB(500 trees, depth 4), MLP(256-128-64, 1000 iter), all seed=42
- KFold(10, shuffle=True, random_state=42)
