# What harmonic phonon calculations capture — and miss — about thermal conductivity

Alexandru Toader

Independent researcher, Buzău, Romania. toader_alexandru@yahoo.com

## Abstract

Harmonic force constants predict lattice thermal conductivity with cross-validated R² = 0.86–0.89 across 751 crystalline solids spanning 4.7 orders of magnitude in κ, using 14 standard phonon descriptors. The plateau persists across linear, ensemble, and neural network models and does not improve with additional engineered features. The remaining variance is primarily correlated with cubic coupling strength |V₃|², which is weakly coupled to harmonic bond stiffness after controlling for atomic mass (partial r = −0.05). The prediction ceiling arises not because models are insufficient, but because anharmonic scattering is approximately decoupled from harmonic structure. Rigid oxides and layered compounds illustrate this decoupling most clearly.

## 1. Introduction

Lattice thermal conductivity κ is a central quantity in thermal management of electronics [1], thermoelectric energy conversion [2], and thermal barrier coating design [3]. First-principles prediction of κ through the phonon Boltzmann transport equation (BTE) with anharmonic (third-order) force constants is now well established [4,5] and agrees with experiment to within 10–20% for many crystalline solids [6]. However, the computational cost of obtaining third-order force constants remains substantial — typically requiring hundreds of supercell calculations per material — which limits the throughput of BTE-based screening [7].

Harmonic (second-order) force constants are considerably cheaper to obtain, requiring only a single set of displaced-atom calculations. From the harmonic dynamical matrix one can extract phonon dispersion relations, sound velocities, Debye temperatures, phonon density of states, and derived quantities such as the three-phonon phase space volume [8]. Several studies have used these harmonic properties to estimate κ, either through analytical models [9,10] or machine-learning approaches [11,12], with reported accuracy varying from R² ≈ 0.6 to R² ≈ 0.9 depending on the feature set, model complexity, and dataset. These studies have focused on improving predictive accuracy. What has received less attention is the characterization of the residual: what physical information is absent from harmonic models, and whether any inexpensive descriptor can recover it. If the remaining ~10% of variance reflects missing physics — specifically information contained in higher-order force constants — then improving harmonic models is fundamentally limited.

We address this question on 751 materials from PhononDB [7] for which both harmonic force constants and BTE-computed κ at 300 K are available. We train models of increasing complexity, characterize the plateau, and correlate the harmonic residual with anharmonic quantities available for a 231-material subset. We find that the residual is primarily correlated with cubic anharmonic strength, which is weakly coupled to harmonic bond stiffness, supporting an approximate factorization of κ into contributions that depend on different derivatives of the interatomic potential.

## 2. Data and methods

### 2.1 Dataset

We use 751 materials from PhononDB [7] that have both harmonic force constants (FORCE_SETS from phonopy [13]) and lattice thermal conductivity computed via the linearized BTE at 300 K (phono3py [5]). Three materials with known BTE convergence artifacts are excluded, as are materials with unresolved sound velocities. Ten additional materials are excluded because the electronegativity contrast feature requires at least two distinct elements with known Pauling electronegativities (B, two Si polymorphs, three Hg compounds, four Yb compounds). The dataset spans six crystal systems: orthorhombic (N = 209), monoclinic (175), cubic (136), trigonal (104), tetragonal (85), and hexagonal (42). [Table 1]

### 2.2 Harmonic features

From the harmonic force constants of each material we extract 14 features:

- **Sound velocities** (3): longitudinal v_LA, transverse v_TA, and the Brillouin-zone-averaged squared velocity sv2 = Σ_q v²(q) / N_q, which captures the full dispersion rather than only the long-wavelength limit.
- **Debye temperature** (1): θ_D computed from the second moment of the phonon density of states.
- **Phase space volume** (1): W, the weighted three-phonon scattering phase space [8], computable from the harmonic dispersion alone.
- **Spectral shape** (5): spectral spread, maximum frequency gap, fraction of sv2 below median frequency, spectral entropy, and DOS bandwidth ratio.
- **Structural** (4): average atomic mass, electronegativity contrast, mean diagonal force constant (fc_mean), and interaction range.

All 14 features are computable from FC2 in seconds per material. The target variable is log₁₀(κ_BTE). As a robustness check, we verify that adding 19 engineered features (Morse anharmonicity estimates, crystal system indicators, inverse group velocity proxies) yields minimal improvement in nonlinear models (§3.1).

### 2.3 Models

We train three model families: (1) Ridge regression (linear baseline), (2) gradient boosting (500 trees, max depth 4), and (3) a neural network (MLPRegressor, layers 256-128-64). All are evaluated by 10-fold cross-validation with a fixed random seed (42). Feature ablation adds feature groups incrementally: sound velocities → Debye temperature → spectral shape → structural descriptors → phase space volume.

### 2.4 Anharmonic data

RTA calculations are available for 240 materials in PhononDB; of these, 231 overlap with the main dataset after exclusion criteria. From the RTA results we extract the BZ-averaged phonon scattering rate Γ, weighted by each mode's contribution to thermal conductivity (C_q v²_q / Σ C_q v²_q) as computed by phono3py [5]. We define the effective cubic coupling strength as |V₃|² ≡ Γ/W, which isolates the coupling contribution from the kinematic phase space. This is a transport-weighted proxy rather than a direct FC3 norm, but captures the component of scattering relevant for thermal transport. These quantities are used for residual analysis only. The ratio κ_BTE/κ_RTA serves as a direct test of whether harmonic features carry any anharmonic information.

## 3. Results

### 3.1 Harmonic prediction accuracy

With 14 features, Ridge regression achieves R² = 0.86 ± 0.04, gradient boosting R² = 0.88 ± 0.02, and the neural network R² = 0.89 ± 0.03 on 751 materials spanning κ from 0.01 to 449 W/mK (Figure 1). Feature ablation on Ridge reveals a clear hierarchy: sound velocities alone yield R² = 0.61; adding Debye temperature gives 0.63; spectral shape brings 0.79; structural descriptors reach 0.83; and including phase space volume W completes the set at 0.86. The marginal contribution of W (+0.02) is context-dependent, consistent with its role as a kinematic complement to sound velocity.

Extending to 33 descriptors (+19 engineered features), Ridge improves to R² = 0.89 and the neural network to R² = 0.90, a gain of less than 0.02 over the 14-feature baseline. A learning curve analysis confirms that the plateau is not data-limited: all three models largely saturate by N ≈ 400, with gains of less than 0.01 between N = 400 and N = 751 (Figure 5).

### 3.2 Characterizing the residual

To determine whether the harmonic residual contains anharmonic information, we test two quantities on the 231-material RTA overlap.

First, we ask whether harmonic features predict the ratio log(κ_BTE/κ_RTA), which isolates the beyond-RTA anharmonic correction. All three models give negative R² (Ridge −0.39, GB −2.12, MLP −1.48), substantially worse than predicting the mean. The anharmonic correction carries no detectable harmonic signal within the tested model classes.

Second, we correlate the cross-validated harmonic residual with the anharmonic scattering rate Γ. Using a 5-feature baseline (sv2, mass, fc_mean, θ_D, W), the residual correlates with log(Γ) at r = −0.54. The full 14-feature model yields r = −0.29 (p < 10⁻⁴) — the reduction reflects removal of harmonic variance, not loss of anharmonic signal. Since W's partial correlation with the residual is negligible (r = −0.01 after mass control), the residual's correlation with Γ is consistent with a dominant contribution from cubic coupling strength |V₃|². [Figure 2]

The residual shows no significant dependence on material class (ANOVA p = 0.07 across ten dominant-element groups) and no significant autocorrelation (Durbin-Watson 1.78–2.16).

### 3.3 Weak coupling between stiffness and asymmetry

The correlation between log(fc_mean) and log(|V₃|²) is r = +0.09 (p = 0.15, n = 231). Controlling for atomic mass gives partial r = −0.05 (p = 0.46). A Spearman rank correlation confirms the same pattern: ρ = +0.17 before mass control, ρ = +0.04 after. At this sample size, correlations |r| > 0.13 would be detectable at the 95% level. [Figure 3]

Conditioning on additional harmonic features (sv2, θ_D) reveals a moderate conditional correlation (partial r = −0.28, p < 10⁻⁴). This is physically expected — at fixed phonon structure, stiffer bonds tend to be more symmetric — but reflects redundancy within the harmonic descriptor space rather than access to anharmonic information: removing fc_mean from the 14-feature model changes the residual's correlation with |V₃|² by less than 0.005. This statistical independence is the structural origin of the observed prediction ceiling: bond stiffness does not predict anharmonic scattering strength.

### 3.4 Material classes at the boundary

Two material classes exhibit systematic deviations that illustrate the stiffness-asymmetry decoupling. These examples are illustrative rather than statistically representative. [Figure 4]

**Rigid oxides** (TiO₂, ReO₃, ZrO₂, SnO₂; n = 4): high bond stiffness but high |V₃|² (1.36 standard deviations above mean, Mann-Whitney p = 0.006). The small sample size limits statistical power, but the direction is consistent: rigid metal-oxygen frameworks with soft tilting modes generate strong anharmonic scattering. The harmonic model overpredicts κ for all four.

**Layered compounds** (GaSe (two polymorphs), GaS, GaTe, WS₂, MoS₂, WSe₂, BiI; n = 8): low average stiffness but low |V₃|² (1.60 standard deviations below mean, p < 0.001). The in-plane bonding is covalent and symmetric, suppressing anharmonic scattering despite low average stiffness. The harmonic model underpredicts κ.

Both deviations survive mass control (rigid oxide p = 0.005, layered p < 0.001).

### 3.5 Oracle bound on the harmonic ceiling

Harmonic features jointly predict |V₃|² at cross-validated R² = 0.70, but adding this prediction to the κ model yields zero lift (ΔR² < 0.001) — the FC2 → |V₃|² channel is already fully exploited. The ceiling is set by the residual variance of |V₃|² that FC2 cannot access. Adding the true |V₃|² residual (an oracle test) improves R² by +0.04 (from 0.79 to 0.83 on the 231-material subset), and no tested structural descriptor correlates with this residual (all |r| < 0.04). Breaking the harmonic ceiling requires explicit third-order force constant information.

## 4. Discussion

### Interpretation

The kinetic expression κ = (1/3) Σ_q C(q) v²(q) τ(q) separates naturally into harmonic quantities (C, v) and anharmonic coupling (τ). In the BZ-averaged form, this separation can be made explicit in a minimal form:

κ ≈ A · sv2 · Γ⁻¹

A two-variable regression gives in-sample R² = 0.78 (cross-validated R² = 0.73 ± 0.17) with coefficients a = +1.11, b = −0.80, consistent with the physical expectation (a ≈ +1, b ≈ −1).

### Tested descriptors that do not bridge the gap

We evaluated several candidate descriptors: Morse-potential Grüneisen parameter (ρ = +0.22 with residual, R² lift = +0.008), inverse group velocity and spectral skewness proxies (not significant), and element-based |V₃|² proxies (cross-validated R² = 0.55 for |V₃|² prediction, κ lift = +0.004). None meaningfully reduced the harmonic residual.

While prior work has focused on improving κ prediction from harmonic descriptors [11,12], the present results suggest that such improvements are fundamentally bounded unless anharmonic information is introduced explicitly.

### Practical implications

For high-throughput screening where ranking matters more than absolute accuracy, harmonic models provide useful κ predictions with median relative error ~33%. For higher precision, anharmonic calculations remain necessary. The R² ≈ 0.89 level offers a practical benchmark: further effort is better directed toward computing FC3 for selected candidates than toward engineering richer harmonic descriptors.

### Structural perspective

The weak coupling between harmonic mode structure and anharmonic scattering rates has a structural interpretation. On lattice models with uniform stiffness, acoustic eigenvectors are invariant to overall stiffness scaling — only eigenfrequencies change. Scattering rates involve third-order coupling projected onto these eigenvectors and constitute independent information. The approximate separability observed here is consistent with this picture. More generally, second- and third-order force constant properties describe different aspects of the potential energy surface (curvature vs. asymmetry) that are not constrained to co-vary.

## 5. Conclusion

Fourteen standard harmonic descriptors predict lattice thermal conductivity with R² = 0.86–0.89 on 751 crystalline solids spanning 4.7 orders of magnitude in κ. The remaining variance is primarily correlated with cubic anharmonic strength, which is weakly coupled to harmonic bond stiffness after controlling for atomic mass. An oracle test confirms that the unpredictable component of |V₃|² is the binding constraint, and no tested inexpensive descriptor accesses it. These findings suggest that the accuracy of harmonic κ models is largely governed by the approximate separability of propagation and scattering physics, and help delineate when anharmonic calculations become necessary.

## Data availability

All harmonic features, BTE thermal conductivities, analysis scripts, and 37 reproducibility tests are publicly available at https://github.com/alex-toader/st_kappa_ceiling (DOI: 10.5281/zenodo.19570129). The underlying force constants are from PhononDB [7].

## References

[1] D.G. Cahill et al., Nanoscale thermal transport, J. Appl. Phys. 93, 793 (2003).
[2] G.J. Snyder and E.S. Toberer, Complex thermoelectric materials, Nat. Mater. 7, 105 (2008).
[3] D.R. Clarke and S.R. Phillpot, Thermal barrier coating materials, Mater. Today 8, 22 (2005).
[4] W. Li et al., ShengBTE: a solver of the Boltzmann transport equation for phonons, Comput. Phys. Commun. 185, 1747 (2014).
[5] A. Togo et al., Distributions of phonon lifetimes in Brillouin zones, Phys. Rev. B 91, 094306 (2015).
[6] L. Lindsay, First principles Peierls-Boltzmann phonon thermal transport, in Nanoscale Energy Transport, IOP, 2020.
[7] A. Togo, L. Chaput, T. Tadano, and I. Tanaka, Implementation strategies in phonopy and phono3py, J. Phys.: Condens. Matter 35, 353001 (2023). PhononDB: https://phonondb.mtl.kyoto-u.ac.jp
[8] L. Lindsay and D.A. Broido, Three-phonon phase space and lattice thermal conductivity in semiconductors, J. Phys.: Condens. Matter 20, 165209 (2008).
[9] G.A. Slack, Nonmetallic crystals with high thermal conductivity, J. Phys. Chem. Solids 34, 321 (1973).
[10] D.T. Morelli and G.A. Slack, High lattice thermal conductivity solids, in High Thermal Conductivity Materials, Springer, 2006.
[11] J. Carrete, W. Li, N. Mingo, S. Wang, and S. Curtarolo, Finding unprecedentedly low-thermal-conductivity half-Heusler semiconductors via high-throughput materials modeling, Phys. Rev. X 4, 011019 (2014).
[12] R. Juneja, G. Yumnam, S. Satsangi, and A.K. Singh, Coupling the high-throughput property map to machine learning for predicting lattice thermal conductivity, Chem. Mater. 31, 5145 (2019).
[13] A. Togo and I. Tanaka, First principles phonon calculations in materials science, Scr. Mater. 108, 1 (2015).
