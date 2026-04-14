# The harmonic ceiling of lattice thermal conductivity prediction

Harmonic force constants predict lattice thermal conductivity with R² = 0.86-0.89 across 751 crystalline solids spanning 4.7 orders of magnitude in kappa. The plateau persists across linear, ensemble, and neural network models. The remaining variance maps onto cubic anharmonic coupling strength, which is approximately decoupled from harmonic structure.

**Paper:** A. Toader, "What harmonic phonon calculations capture -- and miss -- about thermal conductivity" (2026).

## Structure

```
data/           11 CSV files from PhononDB (751 materials, harmonic + anharmonic features)
paper/          main.md, main.tex, main.pdf, make_figures.py, 5 figures
tests/          4 reproducibility test files (40 checks total)
```

## Reproduce

```bash
# Run all tests (requires numpy, scipy, sklearn, pandas)
python tests/test_ceiling.py        # 12/12 - R² plateau, ablation, learning curve
python tests/test_residual.py       # 12/12 - BTE/RTA, residual-Gamma, oracle bound
python tests/test_independence.py   # 10/10 - fc_mean vs |V3|², Spearman, MI, bootstrap
python tests/test_outliers.py       #  6/6  - rigid oxides, layered compounds

# Regenerate figures
python paper/make_figures.py

# Build PDF (requires tectonic or pdflatex)
tectonic paper/main.tex
```

## Key results

| Result | Value |
|--------|-------|
| Ridge R² (14 features) | 0.86 +/- 0.04 |
| Neural network R² (14 features) | 0.89 +/- 0.03 |
| Partial r(fc_mean, \|V3\|² \| mass) | -0.05 (p = 0.46) |
| Oracle ceiling improvement | +0.04 R² |
| FC2 predicts \|V3\|² at R² | 0.70 (but zero kappa lift) |

## Data

All harmonic features and BTE thermal conductivities are derived from [PhononDB](https://phonondb.mtl.kyoto-u.ac.jp) (Togo et al., 2023). RTA scattering rates computed by phono3py.

## License

MIT
