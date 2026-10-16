# What harmonic phonon calculations capture — and miss — about thermal conductivity [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19570128.svg)](https://doi.org/10.5281/zenodo.19570128)

Harmonic force constants predict lattice thermal conductivity with R² = 0.86-0.89 across 749 crystalline solids spanning 4.7 orders of magnitude in kappa. The plateau persists across linear, ensemble, and neural network models, and lies below the central label-noise bound measured against an independent database. The remaining variance correlates with a transport-weighted cubic-coupling proxy.

**Read the paper: [paper/main.pdf](paper/main.pdf)** (also as [Markdown](paper/main.md)). A. Toader, "What harmonic phonon calculations capture -- and miss -- about thermal conductivity" (2026).

## Structure

```
data/           13 CSV files: 12 derived from PhononDB (harmonic + anharmonic features; 749 materials analysed), 1 from Phonix
paper/          main.md, main.tex, main.pdf, make_figures.py, 5 figures
tests/          5 reproducibility test files (55 checks total)
```

## Reproduce

```bash
# Install the analysis stack listed under Software, then run all tests
pip install numpy==2.5.3 scipy==1.18.1 pandas==3.0.5 scikit-learn==1.9.0 matplotlib==3.11.1
python tests/test_ceiling.py        # 12/12 - R² plateau, ablation, learning curve
python tests/test_residual.py       #  6/6  - residual-Gamma, ANOVA, Durbin-Watson, factorization, Γ/W oracle
python tests/test_independence.py   #  8/8  - fc_mean vs |V3|², partial r, Spearman
python tests/test_outliers.py       #  8/8  - rigid oxides, layered compounds, residual signs, degenerate-band sensitivity
python tests/test_robustness.py     # 21/21 - dataset facts, imaginary modes, the RTA table and convergence artifacts, the RTA gap, Phonix covariance, RMSE, residual vs |V₃|², the partial r's interval, tested descriptors, grouped CV, learning curves, RTA-κ oracle, error by range, kNN estimate, Slack, outlier classes, Phonix label noise, ranking

# Regenerate figures
python paper/make_figures.py

# Build PDF (requires tectonic or pdflatex)
tectonic paper/main.tex
```

## Key results

| Result | Value |
|--------|-------|
| Ridge R² (14 features) | 0.86 +/- 0.03 |
| Neural network R² (14 features) | 0.89 +/- 0.03 |
| Oracle: Γ/W scalar / coarse RTA κ | +0.05 / +0.11 R² |
| RMSE (log10 κ), all / 1–10 W/mK | 0.24–0.26 / 0.21 dex |
| Grouped CV (composition, chemical system, prototype) | R² moves ≤ 0.02 |
| Label-noise bound on R² (PhononDB vs Phonix, equal noise) | 0.96 (95 % interval 0.93–0.98) |

## Software

- Data: harmonic features with phonopy 3.4.0; scattering rates with phono3py 4.8.2 (phonopy 4.8.3). The labels are PhononDB's (phonopy 2.47.1, phono3py 3.23.0).
- Analysis, tests and figures: Python 3.14.7, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.0, matplotlib 3.11.1.
- PDF: tectonic 0.15.0.

## Data

All harmonic features and the thermal conductivities (phono3py RTA, the trace average) are derived from PhononDB (A. Togo, NIMS Materials Data Repository, https://mdr.nims.go.jp). The scattering rates Γ are our own phono3py RTA runs (default options, force constants computed from PhononDB's displacement–force datasets, 6×6×6 mesh, 300 K, no isotope scattering, v² weighting); the models use 10-fold cross-validation repeated over ten fold assignments.

`data/kappa_rta_phonondb.csv` holds the label: PhononDB's κ at 300 K (phono3py RTA), its xx, yy, zz components and their average.

`data/phonix_kp_overlap_v1.csv` holds the Phonix κ_p values (Ohnishi et al., npj Comput. Mater. 12, 150 (2026); phonix-db.org) for the materials shared with PhononDB, used only for the label-noise estimate.

`data/degenerate_weights_layered_v1.csv` holds Γ_transport of the eight layered materials of §3.4 from the same phono3py 4.8.2 runs with the default options and with `average_degenerate_weights` on (the tetrahedron weights averaged over degenerate bands); the default column equals `features_scattering_rta_v1.csv`.

## License

MIT
