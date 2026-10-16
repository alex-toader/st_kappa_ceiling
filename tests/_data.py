"""Shared data loading for paper reproducibility tests.

Loads the 749-material main dataset and the 229-material RTA overlap,
with the same pipeline as make_figures.py, and the repeated cross-validation every test uses:
10-fold KFold over the ten fold assignments SEEDS (0–9); a cross-validated number is the mean over the
assignments of its value on each, and the ± of an R² is the standard deviation over all 100 folds.
"""
import os
import re
import numpy as np
import pandas as pd

DATA_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'data'))

KAPPA_ARTIFACTS = {1105, 9497, 4823}
STRUCTURE_MISMATCH = {776016, 625739}       # K2O, Al(OH)3: the κ record and the harmonic record belong to different structures
SEEDS = tuple(range(10))

EN_PAULING = {
    'H':2.20,'Li':0.98,'Be':1.57,'B':2.04,'C':2.55,'N':3.04,'O':3.44,'F':3.98,
    'Na':0.93,'Mg':1.31,'Al':1.61,'Si':1.90,'P':2.19,'S':2.58,'Cl':3.16,
    'K':0.82,'Ca':1.00,'Sc':1.36,'Ti':1.54,'V':1.63,'Cr':1.66,'Mn':1.55,'Fe':1.83,
    'Co':1.88,'Ni':1.91,'Cu':1.90,'Zn':1.65,'Ga':1.81,'Ge':2.01,'As':2.18,'Se':2.55,'Br':2.96,
    'Rb':0.82,'Sr':0.95,'Y':1.22,'Zr':1.33,'Nb':1.60,'Mo':2.16,'Ru':2.20,'Rh':2.28,
    'Pd':2.20,'Ag':1.93,'Cd':1.69,'In':1.78,'Sn':1.96,'Sb':2.05,'Te':2.10,'I':2.66,
    'Cs':0.79,'Ba':0.89,'La':1.10,'Hf':1.30,'Ta':1.50,'W':2.36,'Re':1.90,'Os':2.20,
    'Ir':2.20,'Pt':2.28,'Au':2.54,'Tl':1.62,'Pb':2.33,'Bi':2.02,
    'Ce':1.12,'Pr':1.13,'Nd':1.14,'Sm':1.17,'Eu':1.20,'Gd':1.20,'Tb':1.10,
    'Dy':1.22,'Ho':1.23,'Er':1.24,'Tm':1.25,'Lu':1.27,'Th':1.30,'U':1.38,
}

FEATURES_14 = [
    'log_v_LA', 'log_v_TA', 'log_sv2pm', 'log_mass',
    'theta_D_m2_K', 'spectral_spread', 'freq_gap_max_THz',
    'frac_sv2_below_median', 'spectral_entropy', 'delta_en',
    'dos_cv', 'interaction_range', 'fc_mean', 'log_w',
]

FEATURES_5 = ['log_sv2pm', 'log_mass', 'fc_mean', 'theta_D_m2_K', 'log_w']


def composition(formula):
    """Atom counts of a chemical formula with parentheses: Ba(ErTe2)2 → {'Ba': 1, 'Er': 2, 'Te': 4}. Raises on a formula it cannot read."""
    from collections import Counter
    stack, i = [Counter()], 0
    while i < len(formula):
        c = formula[i]
        if c == '(':
            stack.append(Counter()); i += 1
        elif c == ')':
            n = re.match(r'\d*', formula[i + 1:]).group(0)
            grp = stack.pop()
            for k, v in grp.items():
                stack[-1][k] += v * (int(n) if n else 1)
            i += 1 + len(n)
        else:
            m = re.match(r'([A-Z][a-z]?)(\d*)', formula[i:])
            if not m:
                raise ValueError(f'cannot parse formula {formula!r} at {formula[i:]!r}')
            stack[-1][m.group(1)] += int(m.group(2)) if m.group(2) else 1
            i += len(m.group(0))
    if len(stack) != 1:
        raise ValueError(f'unbalanced parentheses in {formula!r}')
    return dict(stack[0])


def _compute_delta_en(formula):
    tokens = re.findall(r'([A-Z][a-z]?)(\d*)', formula)
    elements = []
    for el, n in tokens:
        if el and el in EN_PAULING:
            count = int(n) if n else 1
            elements.extend([EN_PAULING[el]] * count)
    if len(elements) < 2:
        return np.nan
    return max(elements) - min(elements)


def print_stack():
    """Print the interpreter and library versions the run uses (the paper's numbers were made with Python 3.14.7, numpy 2.5.3, scipy 1.18.1, pandas 3.0.5, scikit-learn 1.9.0)."""
    import sys, scipy, sklearn
    print(f'stack: Python {sys.version.split()[0]}, numpy {np.__version__}, scipy {scipy.__version__}, pandas {pd.__version__}, scikit-learn {sklearn.__version__}')


def load_data():
    paths = {
        'mesh': 'features_mesh_v1.csv', 'fc': 'features_fc_v1.csv',
        'debye': 'debye_v1.csv', 'sv2': 'features_sv2_v1.csv',
        'spectral': 'features_spectral_v1.csv', 'sg': 'spacegroup_phonopy_v1.csv',
        'ps': 'features_phase_space_v1.csv',
    }
    dfs = {}
    for key, fname in paths.items():
        dfs[key] = pd.read_csv(os.path.join(DATA_DIR, fname))
        dfs[key]['mp_id'] = dfs[key]['mp_id'].astype(int)

    kb = pd.read_csv(os.path.join(DATA_DIR, 'kappa_rta_phonondb.csv'))
    kb['mp_id'] = kb['mp_id'].astype(int)

    m = dfs['mesh'][['mp_id', 'material', 'v_LA_m_per_s', 'v_TA_m_per_s',
                      'n_atoms_primitive', 'sound_velocity_unresolved',
                      'dos_width_THz', 'freq_max_THz']].copy()
    m = m.merge(dfs['sv2'][['mp_id', 'sv2_per_mode']], on='mp_id')
    m = m.merge(dfs['fc'][['mp_id', 'avg_mass_AMU', 'fc_mean', 'interaction_range']], on='mp_id')
    m = m.merge(dfs['debye'][['mp_id', 'theta_D_m2_K']], on='mp_id')
    m = m.merge(dfs['spectral'][['mp_id', 'spectral_spread', 'freq_gap_max_THz',
                                  'frac_sv2_below_median', 'spectral_entropy']], on='mp_id')
    m = m.merge(kb[['mp_id', 'k_rta_avg']], on='mp_id')
    m = m.merge(dfs['sg'][['mp_id', 'crystal_system']], on='mp_id')
    m = m.merge(dfs['ps'][['mp_id', 'w_mean']], on='mp_id', how='left')

    m['sound_velocity_unresolved'] = m['sound_velocity_unresolved'].astype(bool)
    m = m[~m.sound_velocity_unresolved]
    m = m.dropna(subset=['k_rta_avg'])
    m = m[~m.mp_id.isin(KAPPA_ARTIFACTS)]
    m = m[~m.mp_id.isin(STRUCTURE_MISMATCH)]

    m['log_k'] = np.log10(m['k_rta_avg'])
    m['log_v_LA'] = np.log10(m['v_LA_m_per_s'])
    m['log_v_TA'] = np.log10(m['v_TA_m_per_s'])
    m['log_mass'] = np.log10(m['avg_mass_AMU'])
    m['log_sv2pm'] = np.log10(m['sv2_per_mode'])
    m['delta_en'] = m['material'].apply(_compute_delta_en)
    m['dos_cv'] = m['dos_width_THz'] / m['freq_max_THz']
    m['log_w'] = np.log10(m['w_mean'].clip(lower=1e-8))
    m = m.dropna(subset=['delta_en'])

    rta = pd.read_csv(os.path.join(DATA_DIR, 'features_scattering_rta_v1.csv'))
    rta['mp_id'] = rta['mp_id'].astype(int)
    rta = rta[rta.kappa_rta < 1000].copy()
    rta['log_krta'] = np.log10(rta['kappa_rta'].clip(lower=0.01))
    rta['log_gamma'] = np.log10(rta['gamma_transport'].clip(lower=1e-8))

    mr = m.merge(rta[['mp_id', 'log_gamma', 'log_krta']], on='mp_id')
    mr['log_V3sq'] = mr['log_gamma'] - mr['log_w']

    return m, mr


def make_ridge():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([('sc', StandardScaler()), ('r', Ridge(alpha=1.0))])


def make_gb():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=500, max_depth=4, random_state=42)


def make_mlp():
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    return Pipeline([('sc', StandardScaler()),
                     ('m', MLPRegressor(hidden_layer_sizes=(256, 128, 64),
                                        max_iter=1000, random_state=42))])


def kfold(seed):
    """One of the ten fold assignments."""
    from sklearn.model_selection import KFold
    return KFold(10, shuffle=True, random_state=seed)


def rcv_r2(make_model, X, y):
    """Repeated 10-fold CV R²: (the mean over SEEDS of the fold-mean R², the sd over all folds, the per-seed fold means)."""
    from sklearn.model_selection import cross_val_score
    folds = [cross_val_score(make_model(), X, y, cv=kfold(s), scoring='r2') for s in SEEDS]
    return float(np.mean([f.mean() for f in folds])), float(np.concatenate(folds).std()), [float(f.mean()) for f in folds]


def rcv_mean(stat):
    """The mean over SEEDS of stat(cv), stat a function of one fold assignment returning a number (or a tuple of numbers)."""
    vals = [stat(kfold(s)) for s in SEEDS]
    return tuple(float(np.mean(v)) for v in zip(*vals)) if isinstance(vals[0], tuple) else float(np.mean(vals))


def partial_corr(x, y, z):
    """Partial correlation of x and y controlling for z (one column or several): (r, two-sided p with n − 2 − k degrees of freedom, k the number of controls)."""
    from scipy.stats import t as t_dist
    from numpy.linalg import lstsq
    z_arr = np.column_stack([z, np.ones(len(x))])
    x_r = x - z_arr @ lstsq(z_arr, x, rcond=None)[0]
    y_r = y - z_arr @ lstsq(z_arr, y, rcond=None)[0]
    r = float(np.corrcoef(x_r, y_r)[0, 1])
    df = len(x) - 2 - (z_arr.shape[1] - 1)
    return r, float(2 * t_dist.sf(abs(r) * np.sqrt(df / (1 - r ** 2)), df))


def at_printed(value, printed):
    """True when `value` rounds to the number as the paper prints it: |value − printed| ≤ half a unit of the printed last digit (a test that carries a printed number fails any other value)."""
    s = str(printed).replace('−', '-').lstrip('+')
    d = len(s.split('.')[1]) if '.' in s else 0
    return abs(value - float(s)) <= 0.5 * 10 ** -d + 1e-12
