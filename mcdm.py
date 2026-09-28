"""Calculs d'aide à la décision multicritère (MCDM).

Formules et notations du cours « Aide à la décision » (Prof. Lamrani Alaoui, EMI) :

- Pondération subjective : AHP (méthode approximative + test de cohérence), BWM (modèle linéaire).
- Pondération objective : entropie, CRITIC.
- Classement : WSM (somme pondérée) et TOPSIS.

Convention : X est la matrice de décision (m alternatives × n critères),
`benefit[j]` vaut True si le critère j est à maximiser, False s'il est à minimiser.
Chaque fonction renvoie un dictionnaire contenant le résultat, les étapes
intermédiaires (pour l'affichage) et une liste d'avertissements en français.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

EPS = 1e-12

# Indice de cohérence aléatoire RI (tableau du cours pour n = 3..10,
# valeurs de Saaty au-delà ; RI = 0 pour n ≤ 2 : une matrice 2×2 est toujours cohérente).
RANDOM_INDEX = {
    1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41,
    9: 1.45, 10: 1.56, 11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}

# Échelle de Saaty (1 à 9) et ses inverses.
SAATY_VALUES = [9, 8, 7, 6, 5, 4, 3, 2, 1, 1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6, 1 / 7, 1 / 8, 1 / 9]


# ---------------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------------

def _as_arrays(X, benefit):
    X = np.asarray(X, dtype=float)
    benefit = np.asarray(benefit, dtype=bool)
    if X.ndim != 2:
        raise ValueError("La matrice de décision doit être à deux dimensions.")
    if X.shape[1] != benefit.size:
        raise ValueError("Le nombre de critères ne correspond pas au nombre de sens (max/min).")
    if np.isnan(X).any():
        raise ValueError("La matrice de décision contient des valeurs manquantes.")
    if not np.isfinite(X).all():
        raise ValueError("La matrice de décision contient des valeurs infinies.")
    return X, benefit


def constant_columns(X) -> np.ndarray:
    """Masque des critères dont toutes les valeurs sont identiques."""
    X = np.asarray(X, dtype=float)
    return np.ptp(X, axis=0) < EPS


def normalize_ratio(X, benefit, names=None):
    """Normalisation linéaire du cours (diapo 57), utilisée par WSM et l'entropie.

    Critère positif : r = x / max(x).  Critère négatif : r = min(x) / x.
    Ces ratios n'ont de sens que pour des valeurs strictement positives ; sinon la
    colonne est normalisée par min-max (même sens de préférence) et un avertissement est émis.
    Renvoie (R, méthode_par_colonne, avertissements).
    """
    X, benefit = _as_arrays(X, benefit)
    m, n = X.shape
    names = names or [f"C{j + 1}" for j in range(n)]
    R = np.zeros_like(X)
    methods, warnings = [], []
    for j in range(n):
        col = X[:, j]
        if (col > 0).all():
            R[:, j] = col / col.max() if benefit[j] else col.min() / col
            methods.append("x / max" if benefit[j] else "min / x")
        else:
            R[:, j] = _minmax_column(col, benefit[j])
            methods.append("min-max")
            warnings.append(
                f"« {names[j]} » contient des valeurs nulles ou négatives : les ratios x/max ou min/x "
                "ne sont pas définis, la colonne est normalisée par min-max."
            )
    return R, methods, warnings


def _minmax_column(col, is_benefit):
    rng = col.max() - col.min()
    if rng < EPS:
        return np.ones_like(col)
    return (col - col.min()) / rng if is_benefit else (col.max() - col) / rng


def rank_scores(scores, higher_is_better=True, decimals=9):
    """Rangs 1..m avec gestion des ex æquo (rang minimal partagé, ex. 1, 2, 2, 4)."""
    s = np.round(np.asarray(scores, dtype=float), decimals)
    key = -s if higher_is_better else s
    ranks = np.empty(len(s), dtype=int)
    for i, v in enumerate(key):
        ranks[i] = 1 + int(np.sum(key < v))
    tied = np.array([np.sum(ranks == r) > 1 for r in ranks])
    return ranks, tied


def spearman(r1, r2) -> float:
    """Corrélation de Spearman entre deux classements (rangs moyens pour les ex æquo)."""
    from scipy.stats import spearmanr

    r1, r2 = np.asarray(r1, dtype=float), np.asarray(r2, dtype=float)
    if np.ptp(r1) == 0 or np.ptp(r2) == 0:
        return float("nan")
    return float(spearmanr(r1, r2).statistic)


# ---------------------------------------------------------------------------
# Pondération subjective : AHP
# ---------------------------------------------------------------------------

def ahp_matrix_from_judgments(n, judgments):
    """Construit la matrice de comparaison par paires à partir du triangle supérieur.

    `judgments[(i, j)]` (i < j) = importance du critère i par rapport au critère j ;
    la réciprocité a_ji = 1 / a_ij est appliquée automatiquement.
    """
    A = np.ones((n, n))
    for (i, j), v in judgments.items():
        v = float(v)
        if v <= 0:
            raise ValueError("Les jugements AHP doivent être strictement positifs.")
        A[i, j] = v
        A[j, i] = 1.0 / v
    return A


def ahp_weights(A):
    """Poids AHP par la méthode approximative du cours (diapos 27-30).

    1) somme de chaque colonne s_j ; 2) division de chaque élément par s_j ;
    3) poids w_i = moyenne de la ligne i ; 4) λmax = Σ s_j w_j,
    CI = (λmax − n)/(n − 1), CR = CI / RI ; cohérent si CR < 0,1.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n):
        raise ValueError("La matrice AHP doit être carrée.")
    if (A <= 0).any():
        raise ValueError("La matrice AHP doit contenir des valeurs strictement positives.")
    warnings = []
    if not np.allclose(A * A.T, 1.0, atol=1e-6):
        warnings.append("La matrice n'est pas réciproque (a_ji ≠ 1/a_ij).")

    col_sums = A.sum(axis=0)
    normalized = A / col_sums
    w = normalized.mean(axis=1)
    lambda_max = float(col_sums @ w)
    if n <= 2:
        ci, ri, cr = 0.0, 0.0, 0.0
    else:
        ci = (lambda_max - n) / (n - 1)
        ri = RANDOM_INDEX.get(n, 1.59)
        cr = ci / ri
        if n > 10:
            warnings.append("Le tableau RI du cours s'arrête à n = 10 : valeurs de Saaty utilisées au-delà.")
    consistent = cr < 0.1

    # Vecteur propre exact (méthode de la puissance), à titre de comparaison.
    eigvals, eigvecs = np.linalg.eig(A)
    k = int(np.argmax(eigvals.real))
    exact = np.abs(eigvecs[:, k].real)
    exact = exact / exact.sum()

    return {
        "weights": w,
        "matrix": A,
        "col_sums": col_sums,
        "normalized": normalized,
        "lambda_max": lambda_max,
        "CI": ci,
        "RI": ri,
        "CR": cr,
        "consistent": consistent,
        "exact_weights": exact,
        "exact_lambda_max": float(eigvals.real[k]),
        "warnings": warnings,
    }


def ahp_rank(alt_matrices, weights, names=None):
    """Classement AHP des alternatives (diapos 31-32).

    `alt_matrices[j]` : matrice de comparaison par paires des alternatives selon le critère j.
    Priorités locales w_ij obtenues comme pour les critères (même test de cohérence), puis
    priorités finales P_i = Σ_j w_j · w_ij ; la plus élevée est la meilleure.
    """
    w = _check_weights(weights, len(alt_matrices))
    names = names or [f"C{j + 1}" for j in range(len(alt_matrices))]
    locals_ = [ahp_weights(A) for A in alt_matrices]
    P = np.column_stack([r["weights"] for r in locals_])
    scores = P @ w
    ranks, tied = rank_scores(scores, higher_is_better=True)
    inconsistent = [names[j] for j, r in enumerate(locals_) if not r["consistent"]]
    return {
        "scores": scores,
        "ranks": ranks,
        "tied": tied,
        "local": P,
        "local_results": locals_,
        "inconsistent": inconsistent,
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# Pondération subjective : BWM
# ---------------------------------------------------------------------------

def bwm_weights(best, worst, a_bo, a_ow):
    """BWM linéaire (diapos 39-44), résolu par programmation linéaire.

    Variables x = (w_1, …, w_n, ξ).  min ξ  s.c. pour tout j :
        |w_B − a_Bj w_j| ≤ ξ,   |w_j − a_jW w_W| ≤ ξ,   Σ w_j = 1,  w_j ≥ 0,  ξ ≥ 0.
    """
    a_bo = np.asarray(a_bo, dtype=float)
    a_ow = np.asarray(a_ow, dtype=float)
    n = a_bo.size
    if a_ow.size != n:
        raise ValueError("Les vecteurs BO et OW doivent avoir la même taille.")
    if n < 2:
        raise ValueError("BWM nécessite au moins deux critères.")
    if best == worst:
        raise ValueError("Le meilleur critère et le pire critère doivent être différents.")
    if ((a_bo < 1) | (a_bo > 9) | (a_ow < 1) | (a_ow > 9)).any():
        raise ValueError("Les notes BWM doivent être comprises entre 1 et 9.")

    warnings = []
    if a_bo[best] != 1 or a_ow[worst] != 1:
        warnings.append("Contrôle du cours non respecté : a_BB et a_WW doivent valoir 1.")
    if a_bo[worst] < a_bo.max():
        warnings.append(
            "Contrôle du cours non respecté : la note la plus élevée du vecteur BO devrait porter sur le pire critère W."
        )
    if a_ow[best] < a_ow.max():
        warnings.append("La note la plus élevée du vecteur OW devrait porter sur le meilleur critère B.")
    if a_bo[worst] != a_ow[best]:
        warnings.append(
            f"a_BW (vecteur BO) = {a_bo[worst]:g} mais a_BW (vecteur OW) = {a_ow[best]:g} : "
            "ces deux notes décrivent la même comparaison B/W et devraient être égales."
        )

    rows = []
    for j in range(n):
        # w_B − a_Bj w_j − ξ ≤ 0   et   −w_B + a_Bj w_j − ξ ≤ 0
        r = np.zeros(n + 1); r[best] += 1; r[j] -= a_bo[j]; r[n] = -1; rows.append(r)
        r = np.zeros(n + 1); r[best] -= 1; r[j] += a_bo[j]; r[n] = -1; rows.append(r)
        # w_j − a_jW w_W − ξ ≤ 0   et   −w_j + a_jW w_W − ξ ≤ 0
        r = np.zeros(n + 1); r[j] += 1; r[worst] -= a_ow[j]; r[n] = -1; rows.append(r)
        r = np.zeros(n + 1); r[j] -= 1; r[worst] += a_ow[j]; r[n] = -1; rows.append(r)
    A_ub = np.array(rows)
    b_ub = np.zeros(len(rows))
    A_eq = np.r_[np.ones(n), 0.0].reshape(1, -1)
    b_eq = np.array([1.0])
    c = np.r_[np.zeros(n), 1.0]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=[(0, None)] * (n + 1), method="highs")
    if not res.success:
        raise RuntimeError(f"Le programme linéaire BWM n'a pas pu être résolu : {res.message}")

    w = np.clip(res.x[:n], 0, None)
    w = w / w.sum()
    xi = float(res.x[n])

    # Cohérence des jugements : idéalement a_Bj × a_jW = a_BW pour tout j.
    a_bw = a_bo[worst]
    products = a_bo * a_ow
    return {
        "weights": w,
        "xi": xi,
        "A_ub": A_ub,
        "b_ub": b_ub,
        "c": c,
        "a_bw": a_bw,
        "products": products,
        "max_gap": float(np.max(np.abs(products - a_bw))),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Pondération objective : entropie
# ---------------------------------------------------------------------------

def entropy_weights(X, benefit, names=None):
    """Méthode de l'entropie (diapos 46-47).

    D normalisée (ratios du cours), p_ij = d_ij / Σ_i d_ij, E_j = −k Σ_i p_ij ln p_ij avec k = 1/ln m,
    w_j = (1 − E_j) / (n − Σ_k E_k).
    """
    X, benefit = _as_arrays(X, benefit)
    m, n = X.shape
    names = names or [f"C{j + 1}" for j in range(n)]
    if m < 2:
        raise ValueError("L'entropie nécessite au moins deux alternatives.")
    D, methods, warnings = normalize_ratio(X, benefit, names)
    col_sums = D.sum(axis=0)
    P = np.zeros_like(D)
    E = np.ones(n)
    for j in range(n):
        if col_sums[j] < EPS:
            warnings.append(f"« {names[j]} » : somme nulle après normalisation, entropie fixée à 1 (poids nul).")
            continue
        P[:, j] = D[:, j] / col_sums[j]
        p = P[:, j]
        with np.errstate(divide="ignore", invalid="ignore"):
            plogp = np.where(p > 0, p * np.log(p), 0.0)  # convention 0 × ln 0 = 0
        E[j] = -plogp.sum() / np.log(m)
    E = np.clip(E, 0.0, 1.0)
    for j in np.where(constant_columns(X))[0]:
        warnings.append(f"« {names[j]} » est constant : il ne distingue pas les alternatives (poids nul).")
    diversification = 1 - E
    denom = n - E.sum()
    if denom < EPS:
        warnings.append("Aucun critère ne discrimine les alternatives : poids égaux attribués par défaut.")
        w = np.full(n, 1 / n)
    else:
        w = diversification / denom
    return {
        "weights": w,
        "normalized": D,
        "normalization": methods,
        "P": P,
        "entropy": E,
        "diversification": diversification,
        "k": 1 / np.log(m),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Pondération objective : CRITIC
# ---------------------------------------------------------------------------

def critic_weights(X, benefit, names=None):
    """Méthode CRITIC (diapos 48-52).

    1) normalisation min-max selon le sens ; 2) corrélation de Pearson ρ_jk ;
    3) σ_j (écart type, dénominateur m − 1) et C_j = σ_j Σ_k (1 − |ρ_jk|) ; 4) w_j = C_j / Σ C_j.
    """
    X, benefit = _as_arrays(X, benefit)
    m, n = X.shape
    names = names or [f"C{j + 1}" for j in range(n)]
    warnings = []
    if m < 3:
        warnings.append(
            "Avec moins de 3 alternatives, toutes les corrélations valent ±1 : CRITIC n'est pas informatif."
        )
    const = constant_columns(X)
    R = np.zeros_like(X)
    for j in range(n):
        if const[j]:
            warnings.append(
                f"« {names[j]} » est constant : écart type nul, corrélations non définies (fixées à 0), poids nul."
            )
            continue
        R[:, j] = _minmax_column(X[:, j], benefit[j])
    sigma = R.std(axis=0, ddof=1) if m > 1 else np.zeros(n)
    means = R.mean(axis=0)
    centered = R - means
    rho = np.eye(n)
    for j in range(n):
        for k in range(j + 1, n):
            den = np.sqrt((centered[:, j] ** 2).sum() * (centered[:, k] ** 2).sum())
            val = 0.0 if den < EPS else float((centered[:, j] * centered[:, k]).sum() / den)
            rho[j, k] = rho[k, j] = val
    independence = (1 - np.abs(rho)).sum(axis=1)
    C = sigma * independence
    if C.sum() < EPS:
        warnings.append("Tous les indices C_j sont nuls : poids égaux attribués par défaut.")
        w = np.full(n, 1 / n)
    else:
        w = C / C.sum()
    return {
        "weights": w,
        "normalized": R,
        "means": means,
        "sigma": sigma,
        "rho": rho,
        "independence": independence,
        "C": C,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Classement : WSM
# ---------------------------------------------------------------------------

def wsm(X, benefit, weights, names=None):
    """Méthode des sommes pondérées, WSM (diapos 55-58) : Q_i = Σ_j w_j r_ij, la plus grande est la meilleure."""
    X, benefit = _as_arrays(X, benefit)
    w = _check_weights(weights, X.shape[1])
    R, methods, warnings = normalize_ratio(X, benefit, names)
    Q = R @ w
    ranks, tied = rank_scores(Q, higher_is_better=True)
    return {
        "scores": Q,
        "ranks": ranks,
        "tied": tied,
        "normalized": R,
        "normalization": methods,
        "weighted": R * w,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Classement : TOPSIS
# ---------------------------------------------------------------------------

def topsis(X, benefit, weights, names=None):
    """TOPSIS (diapos 62-66).

    r_ij = x_ij / √(Σ_i x_ij²), v_ij = w_j r_ij, solutions idéales I+ et I−,
    séparations S_i+ et S_i−, coefficient de proximité RC_i = S_i− / (S_i− + S_i+).
    """
    X, benefit = _as_arrays(X, benefit)
    m, n = X.shape
    names = names or [f"C{j + 1}" for j in range(n)]
    w = _check_weights(weights, n)
    warnings = []
    norms = np.sqrt((X ** 2).sum(axis=0))
    R = np.zeros_like(X)
    for j in range(n):
        if norms[j] < EPS:
            warnings.append(f"« {names[j]} » ne contient que des zéros : colonne normalisée à 0.")
        else:
            R[:, j] = X[:, j] / norms[j]
    if (X < 0).any():
        warnings.append("La matrice contient des valeurs négatives : la normalisation vectorielle reste calculable "
                        "mais s'interprète moins bien ; vérifiez les données.")
    V = R * w
    ideal_pos = np.where(benefit, V.max(axis=0), V.min(axis=0))
    ideal_neg = np.where(benefit, V.min(axis=0), V.max(axis=0))
    s_pos = np.sqrt(((V - ideal_pos) ** 2).sum(axis=1))
    s_neg = np.sqrt(((V - ideal_neg) ** 2).sum(axis=1))
    total = s_pos + s_neg
    rc = np.where(total < EPS, 0.5, s_neg / np.where(total < EPS, 1, total))
    if (total < EPS).any():
        warnings.append("Certaines alternatives sont à égale distance nulle des deux idéaux : RC fixé à 0,5.")
    ranks, tied = rank_scores(rc, higher_is_better=True)
    return {
        "scores": rc,
        "ranks": ranks,
        "tied": tied,
        "normalized": R,
        "weighted": V,
        "ideal_pos": ideal_pos,
        "ideal_neg": ideal_neg,
        "s_pos": s_pos,
        "s_neg": s_neg,
        "warnings": warnings,
    }


def _check_weights(weights, n):
    w = np.asarray(weights, dtype=float)
    if w.size != n:
        raise ValueError("Le nombre de poids ne correspond pas au nombre de critères.")
    if (w < 0).any() or not np.isfinite(w).all():
        raise ValueError("Les poids doivent être positifs et finis.")
    if w.sum() < EPS:
        raise ValueError("La somme des poids est nulle.")
    return w / w.sum()
