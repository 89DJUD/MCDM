"""Application Streamlit d'aide à la décision multicritère (AHP, BWM, entropie, CRITIC → WSM, TOPSIS)."""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import mcdm
from examples import BLANK, BLANK_PROBLEM, EXAMPLES, MAX, MIN

st.set_page_config(page_title="Aide à la décision multicritère", layout="wide")

WEIGHT_METHODS = ["AHP", "BWM", "Entropie", "CRITIC"]
RANK_METHODS = ["WSM", "TOPSIS", "AHP"]



# ---------------------------------------------------------------------------
# État de session
# ---------------------------------------------------------------------------

def snap_saaty(v):
    return min(mcdm.SAATY_VALUES, key=lambda s: abs(np.log(s) - np.log(max(v, 1e-9))))


def load_example(name: str) -> None:
    """Charge un exemple (ou un problème vierge) : structure, matrice, jugements AHP et BWM."""
    ex = BLANK_PROBLEM if name == BLANK else EXAMPLES[name]
    ss = st.session_state
    crits = [c for c, _ in ex["criteria"]]
    ss.problem = ex["problem"]
    ss.crit_list = [list(c) for c in ex["criteria"]]
    ss.alt_list = list(ex["alternatives"])
    ss.cells = {} if ex["matrix"] is None else {
        (a, c): float(ex["matrix"][i][j])
        for i, a in enumerate(ex["alternatives"])
        for j, c in enumerate(crits)
    }
    ss.ahp_store = {(crits[i], crits[j]): float(v) for (i, j), v in ex["ahp"].items()}
    b = ex["bwm"]
    ss.bwm_store = {
        "best": crits[b["best"]],
        "worst": crits[b["worst"]],
        "bo": dict(zip(crits, b["bo"])),
        "ow": dict(zip(crits, b["ow"])),
    }
    alts_ex = ex["alternatives"]
    ss.ahp_alt_store = {
        (crits[j], alts_ex[a], alts_ex[b]): float(v)
        for j, pairs in ex.get("ahp_alt", {}).items()
        for (a, b), v in pairs.items()
    }
    if ex["matrix"] is not None and "ahp_alt" not in ex:
        # Comparaisons AHP des alternatives déduites des rapports de la matrice (arrondis à l'échelle 1-9).
        for j, (c, sens) in enumerate(ex["criteria"]):
            for a in range(len(alts_ex)):
                for b in range(a + 1, len(alts_ex)):
                    xa, xb = ex["matrix"][a][j], ex["matrix"][b][j]
                    ratio = (xa / xb if sens == MAX else xb / xa) if xa > 0 and xb > 0 else 1.0
                    ss.ahp_alt_store[(c, alts_ex[a], alts_ex[b])] = snap_saaty(min(max(ratio, 1 / 9), 9))
    if "methods" in ex:
        ss.w_method, ss.r_method = ex["methods"]
    ss.force_ahp_alt = False
    ss.version = ss.get("version", 0) + 1
    ss.struct = None
    ss.force_ahp = False
    # Oublie l'état des widgets de jugement pour qu'ils se réinitialisent depuis les stores.
    for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith(("_ahp::", "_ahpa::", "_bwm", "_c_", "_a_", "_ncrit", "_nalt"))]:
        del ss[k]


if "version" not in st.session_state:
    load_example(BLANK)
ss = st.session_state
ss.setdefault("ahp_alt_store", {})


def fmt(x, d=4):
    return f"{x:.{d}f}".replace(".", ",")


def saaty_label(v, ci, cj):
    if abs(v - 1) < 1e-9:
        return "Égale importance"
    if v > 1:
        return f"{ci} ×{round(v):d}"
    return f"{cj} ×{round(1 / v):d}"


# ---------------------------------------------------------------------------
# En-tête et exemples
# ---------------------------------------------------------------------------

st.title("Aide à la décision multicritère")

with st.sidebar:
    st.header("Méthodes")
    w_method = st.radio(
        "1. Pondération des critères",
        WEIGHT_METHODS,
        key="w_method",
        captions=[
            "Subjective : comparaisons par paires",
            "Subjective : meilleur et pire critère",
            "Objective : calculée depuis la matrice",
            "Objective : dispersion et corrélations",
        ],
    )
    r_method = st.radio(
        "2. Classement des alternatives",
        RANK_METHODS,
        key="r_method",
        captions=["Somme pondérée", "Distance aux solutions idéales", "Comparaisons par paires des alternatives"],
    )
    st.caption(f"Combinaison active : **{w_method} + {r_method}**. Toutes les combinaisons sont comparées à l'étape 6.")
st.caption(
    "Pondération des critères par AHP, BWM, entropie ou CRITIC, puis classement des alternatives par WSM ou TOPSIS, "
    "avec les formules du cours « Aide à la décision » (EMI)."
)

with st.container(border=True):
    st.subheader("1. Problème")
    c1, c2 = st.columns([2, 3], vertical_alignment="bottom")
    ss.problem = c1.text_input("Nom du problème", value=ss.problem)
    with c2.container(horizontal=True, vertical_alignment="bottom"):
        chosen = st.selectbox("Point de départ", [BLANK, *EXAMPLES], key="example_choice",
                              help="Partez d'une matrice vide ou chargez un exercice du cours pour tester.")
        st.button("Charger", icon=":material/restart_alt:", on_click=load_example, args=(chosen,))

# ---------------------------------------------------------------------------
# 2. Critères et alternatives
# ---------------------------------------------------------------------------

SENS_LABELS = {MAX: ":material/trending_up: Maximiser", MIN: ":material/trending_down: Minimiser"}

with st.container(border=True):
    st.subheader("2. Critères et alternatives")
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown("**Critères**")
        st.caption(
            "Maximiser : plus la valeur est élevée, mieux c'est (qualité, rendement…). "
            "Minimiser : plus la valeur est faible, mieux c'est (coût, délai, risque…)."
        )
        if "_ncrit" not in ss:
            ss["_ncrit"] = len(ss.crit_list)
        nc = st.number_input("Nombre de critères", min_value=2, max_value=15, step=1, key="_ncrit")
        while len(ss.crit_list) < nc:
            ss.crit_list.append([f"Critère {len(ss.crit_list) + 1}", MAX])
        for i in range(nc):
            kn, ks = f"_c_name_{i}", f"_c_sens_{i}"
            if kn not in ss:
                ss[kn] = ss.crit_list[i][0]
            if ks not in ss:
                ss[ks] = ss.crit_list[i][1]
            c1, c2 = st.columns([3, 2], vertical_alignment="bottom")
            name = c1.text_input(f"Nom du critère {i + 1}", key=kn, placeholder="ex. Prix, Qualité, Délai…")
            sens = c2.segmented_control(
                f"Objectif du critère {i + 1}", [MAX, MIN], key=ks, required=True,
                format_func=SENS_LABELS.get, label_visibility="collapsed",
            )
            ss.crit_list[i] = [name, sens or MAX]

    with right:
        st.markdown("**Alternatives**")
        st.caption("Les options à comparer (produits, fournisseurs, projets…).")
        if "_nalt" not in ss:
            ss["_nalt"] = len(ss.alt_list)
        na = st.number_input("Nombre d'alternatives", min_value=2, max_value=50, step=1, key="_nalt")
        while len(ss.alt_list) < na:
            ss.alt_list.append(f"Alternative {len(ss.alt_list) + 1}")
        for i in range(na):
            ka = f"_a_name_{i}"
            if ka not in ss:
                ss[ka] = ss.alt_list[i]
            ss.alt_list[i] = st.text_input(f"Nom de l'alternative {i + 1}", key=ka)

    crit_rows = [(str(n).strip(), sv) for n, sv in ss.crit_list[:nc]]
    crits = [c for c, _ in crit_rows]
    benefit = np.array([sv == MAX for _, sv in crit_rows])
    alts = [str(a).strip() for a in ss.alt_list[:na]]

    errors = []
    empty_c = [str(i + 1) for i, c in enumerate(crits) if not c]
    empty_a = [str(i + 1) for i, a in enumerate(alts) if not a]
    if empty_c:
        errors.append(f"Donnez un nom au(x) critère(s) n° {', '.join(empty_c)}.")
    if empty_a:
        errors.append(f"Donnez un nom à la (aux) alternative(s) n° {', '.join(empty_a)}.")
    dup_c = sorted({c for c in crits if crits.count(c) > 1})
    dup_a = sorted({a for a in alts if alts.count(a) > 1})
    if dup_c:
        errors.append(f"Noms de critères en double : {', '.join(dup_c)}.")
    if dup_a:
        errors.append(f"Noms d'alternatives en double : {', '.join(dup_a)}.")
    if errors:
        for e in errors:
            st.error(e, icon=":material/error:")
        st.stop()

# ---------------------------------------------------------------------------
# 3. Matrice de décision
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.subheader("3. Matrice de décision")
    st.caption("x_ij : performance de l'alternative A_i selon le critère C_j. Saisissez des valeurs numériques.")
    struct = (tuple(alts), tuple(crits), ss.version)
    if ss.struct != struct:
        ss.matrix_base = pd.DataFrame(
            [[ss.cells.get((a, c), np.nan) for c in crits] for a in alts], index=alts, columns=crits, dtype=float
        )
        ss.struct = struct
    matrix_key = "matrix_" + hashlib.md5(repr(struct).encode()).hexdigest()
    sens_help = {c: ("À maximiser" if b else "À minimiser") for c, b in zip(crits, benefit)}
    matrix_df = st.data_editor(
        ss.matrix_base,
        key=matrix_key,
        column_config={
            c: st.column_config.NumberColumn(f"{c} ({'↑' if b else '↓'})", help=sens_help[c])
            for c, b in zip(crits, benefit)
        },
    )
    # Lecture par position : l'éditeur peut renvoyer des libellés d'un état antérieur lors d'un changement de structure.
    if matrix_df.shape == (len(alts), len(crits)):
        vals = matrix_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    else:
        vals = matrix_df.reindex(index=alts, columns=crits).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    matrix_df = pd.DataFrame(vals, index=alts, columns=crits)
    for i, a in enumerate(alts):
        for j, c in enumerate(crits):
            ss.cells[(a, c)] = vals[i, j]

    X = vals
    missing = [f"{a} / {c}" for i, a in enumerate(alts) for j, c in enumerate(crits) if np.isnan(X[i, j])]
    matrix_needed = not (r_method == "AHP" and w_method in ("AHP", "BWM"))
    if missing and not matrix_needed:
        st.info(
            f"Avec {w_method} + AHP, la matrice est facultative : les alternatives sont classées par vos comparaisons "
            "par paires (étape 5). Remplie, elle sert à préremplir ces comparaisons.",
            icon=":material/info:",
        )
        X = None
    elif len(missing) == X.size:
        st.info(
            "Saisissez vos valeurs dans la matrice ci-dessus (une valeur numérique par case). "
            "Renommez d'abord critères et alternatives à l'étape 2 si nécessaire.",
            icon=":material/edit:",
        )
        st.stop()
    elif missing:
        st.error(
            f"{len(missing)} valeur(s) manquante(s) : {', '.join(missing[:8])}{' …' if len(missing) > 8 else ''}. "
            "Complétez la matrice pour poursuivre.",
            icon=":material/error:",
        )
        st.stop()
    const = mcdm.constant_columns(X) if X is not None else np.zeros(len(crits), dtype=bool)
    if const.any():
        st.warning(
            "Critère(s) constant(s) : " + ", ".join(c for c, k in zip(crits, const) if k)
            + ". Ils ne permettent pas de départager les alternatives.",
            icon=":material/warning:",
        )
    if X is not None and (X <= 0).any():
        st.info(
            "La matrice contient des valeurs nulles ou négatives : les colonnes concernées seront normalisées par "
            "min-max pour WSM et l'entropie (les ratios du cours exigent des valeurs strictement positives).",
            icon=":material/info:",
        )

n, m = len(crits), len(alts)


# ---------------------------------------------------------------------------
# Calcul des poids (réutilisé par la section et par la comparaison)
# ---------------------------------------------------------------------------

def ahp_result():
    judg = {}
    for i in range(n):
        for j in range(i + 1, n):
            v = ss.ahp_store.get((crits[i], crits[j]))
            if v is None and (crits[j], crits[i]) in ss.ahp_store:
                v = 1 / ss.ahp_store[(crits[j], crits[i])]
            judg[(i, j)] = snap_saaty(v if v else 1.0)
    return mcdm.ahp_weights(mcdm.ahp_matrix_from_judgments(n, judg))


def bwm_ensure_valid():
    b = ss.bwm_store
    if b.get("best") not in crits:
        b["best"] = crits[0]
    if b.get("worst") not in crits:
        b["worst"] = crits[-1]


def bwm_result():
    bwm_ensure_valid()
    b = ss.bwm_store
    bi, wi = crits.index(b["best"]), crits.index(b["worst"])
    bo = [1 if c == b["best"] else int(b["bo"].get(c, 1)) for c in crits]
    ow = [1 if c == b["worst"] else int(b["ow"].get(c, 1)) for c in crits]
    return mcdm.bwm_weights(bi, wi, bo, ow)


def compute_weights(method):
    """Renvoie (poids ou None, résultat détaillé ou message d'erreur)."""
    try:
        if method == "AHP":
            r = ahp_result()
            ok = r["consistent"] or ss.get("force_ahp", False)
            return (r["weights"] if ok else None), r
        if method == "BWM":
            r = bwm_result()
            return r["weights"], r
        if X is None and method in ("Entropie", "CRITIC"):
            return None, "la matrice de décision est incomplète"
        if method == "Entropie":
            r = mcdm.entropy_weights(X, benefit, crits)
            return r["weights"], r
        r = mcdm.critic_weights(X, benefit, crits)
        return r["weights"], r
    except (ValueError, RuntimeError) as exc:
        return None, str(exc)


def ahp_alt_matrix(c):
    """Matrice de comparaison des alternatives pour le critère c, depuis les jugements mémorisés."""
    judg = {}
    for i in range(m):
        for j in range(i + 1, m):
            v = ss.ahp_alt_store.get((c, alts[i], alts[j]))
            if v is None and (c, alts[j], alts[i]) in ss.ahp_alt_store:
                v = 1 / ss.ahp_alt_store[(c, alts[j], alts[i])]
            judg[(i, j)] = snap_saaty(v if v else 1.0)
    return mcdm.ahp_matrix_from_judgments(m, judg)


def rank(method, w):
    if method == "AHP":
        return mcdm.ahp_rank([ahp_alt_matrix(c) for c in crits], w, crits)
    return (mcdm.wsm if method == "WSM" else mcdm.topsis)(X, benefit, w, crits)


def prefill_ahp_alt_from_matrix():
    """Préremplit les comparaisons des alternatives par les rapports de la matrice (arrondis à l'échelle 1-9)."""
    for j, c in enumerate(crits):
        for i in range(m):
            for k in range(i + 1, m):
                a, b = ss.cells.get((alts[i], c)), ss.cells.get((alts[k], c))
                if a is None or b is None or np.isnan(a) or np.isnan(b) or a <= 0 or b <= 0:
                    v = 1.0
                else:
                    v = snap_saaty(min(max(a / b if benefit[j] else b / a, 1 / 9), 9))
                ss.ahp_alt_store[(c, alts[i], alts[k])] = v
    for k in [k for k in ss.keys() if isinstance(k, str) and k.startswith("_ahpa::")]:
        del ss[k]


def show_warnings(res):
    for w in res.get("warnings", []):
        st.warning(w, icon=":material/warning:")


def weights_table(w):
    return pd.DataFrame({"Critère": crits, "Poids": w, "Sens": ["Max" if b else "Min" for b in benefit]})


# ---------------------------------------------------------------------------
# 4. Pondération
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.subheader("4. Pondération des critères")
    st.caption(f"Méthode choisie dans le panneau latéral : **{w_method}**.")

    if w_method == "AHP":
        st.markdown(
            "Comparez chaque paire de critères sur l'échelle de Saaty (1 = importance égale, 9 = extrêmement plus "
            "important). Placez le curseur du côté du critère le plus important."
        )
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        cols = st.columns(2)
        for k, (i, j) in enumerate(pairs):
            ci, cj = crits[i], crits[j]
            key = f"_ahp::{ci}::{cj}"
            if key not in ss:
                v = ss.ahp_store.get((ci, cj))
                if v is None and (cj, ci) in ss.ahp_store:
                    v = 1 / ss.ahp_store[(cj, ci)]
                ss[key] = snap_saaty(v or 1.0)
            with cols[k % 2]:
                ss.ahp_store[(ci, cj)] = st.select_slider(
                    f"{ci} ↔ {cj}", options=mcdm.SAATY_VALUES, key=key,
                    format_func=lambda v, ci=ci, cj=cj: saaty_label(v, ci, cj),
                )
        st.caption(f"{len(pairs)} comparaisons pour {n} critères : n(n − 1)/2.")

    elif w_method == "BWM":
        bwm_ensure_valid()
        b = ss.bwm_store
        st.markdown(
            "Désignez le critère le plus important (B) et le moins important (W), puis notez de 1 à 9 : "
            "« combien de fois B est plus important que chaque critère » (vecteur BO) et "
            "« combien de fois chaque critère est plus important que W » (vecteur OW)."
        )
        c1, c2 = st.columns(2)
        if "_bwm_best" not in ss:
            ss["_bwm_best"] = b["best"]
        if "_bwm_worst" not in ss:
            ss["_bwm_worst"] = b["worst"]
        if ss["_bwm_best"] not in crits:
            ss["_bwm_best"] = b["best"]
        if ss["_bwm_worst"] not in crits:
            ss["_bwm_worst"] = b["worst"]
        b["best"] = c1.selectbox("Meilleur critère (B)", crits, key="_bwm_best")
        b["worst"] = c2.selectbox("Pire critère (W)", crits, key="_bwm_worst")
        if b["best"] == b["worst"]:
            st.error("Le meilleur et le pire critère doivent être différents.", icon=":material/error:")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Best-to-Others** : {b['best']} par rapport à…")
            for c in crits:
                if c == b["best"]:
                    st.caption(f"{c} : a_BB = 1")
                    continue
                key = f"_bwm_bo::{c}"
                if key not in ss:
                    ss[key] = int(min(max(b["bo"].get(c, 1), 1), 9))
                b["bo"][c] = st.number_input(c, min_value=1, max_value=9, step=1, key=key)
        with c2:
            st.markdown(f"**Others-to-Worst** : … par rapport à {b['worst']}")
            for c in crits:
                if c == b["worst"]:
                    st.caption(f"{c} : a_WW = 1")
                    continue
                key = f"_bwm_ow::{c}"
                if key not in ss:
                    ss[key] = int(min(max(b["ow"].get(c, 1), 1), 9))
                b["ow"][c] = st.number_input(c, min_value=1, max_value=9, step=1, key=key)
        st.caption(f"{2 * n - 3} comparaisons pour {n} critères : 2n − 3 (contre {n * (n - 1) // 2} pour AHP).")

    weights, wres = compute_weights(w_method)

    if isinstance(wres, str):
        st.error(wres, icon=":material/error:")
    else:
        show_warnings(wres)
        st.markdown("#### Poids obtenus")
        wv = wres["weights"]
        wt = weights_table(wv)
        c1, c2 = st.columns([2, 3])
        c1.dataframe(
            wt, hide_index=True,
            column_config={"Poids": st.column_config.ProgressColumn("Poids", format="%.4f", min_value=0.0, max_value=1.0)},
        )
        c2.bar_chart(wt, x="Critère", y="Poids", horizontal=True, sort="-Poids", height=max(160, 40 * n))

        if w_method == "AHP":
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("λmax", fmt(wres["lambda_max"]))
            k2.metric("CI", fmt(wres["CI"]))
            k3.metric("RI", fmt(wres["RI"], 2))
            k4.metric("CR", fmt(wres["CR"]), help="La matrice est cohérente si CR < 0,1.")
            if n <= 2:
                st.info("Avec 2 critères, la matrice est toujours cohérente (CR non applicable).", icon=":material/info:")
            elif wres["consistent"]:
                st.success(f"CR = {fmt(wres['CR'])} < 0,1 : les jugements sont cohérents, les poids peuvent être conservés.",
                           icon=":material/check_circle:")
            else:
                A, w_ = wres["matrix"], wres["weights"]
                dev = [
                    (max(A[i, j] * w_[j] / w_[i], w_[i] / (A[i, j] * w_[j])), crits[i], crits[j])
                    for i in range(n) for j in range(i + 1, n)
                ]
                worst_pairs = ", ".join(f"{a} ↔ {b_}" for _, a, b_ in sorted(dev, reverse=True)[:3])
                st.error(
                    f"CR = {fmt(wres['CR'])} ≥ 0,1 : les jugements sont incohérents. Selon le cours, il faut les réviser "
                    f"avant de poursuivre. Paires les plus éloignées des poids obtenus : {worst_pairs}.",
                    icon=":material/error:",
                )
                st.checkbox("Utiliser ces poids malgré l'incohérence", key="force_ahp")
                if ss.force_ahp:
                    weights = wv
            with st.expander("Étapes de calcul AHP", icon=":material/calculate:"):
                st.markdown("**Matrice de comparaison par paires** (a_ij = importance de C_i par rapport à C_j) et sommes s_j")
                mat = pd.DataFrame(wres["matrix"], index=crits, columns=crits)
                mat.loc["Somme s_j"] = wres["col_sums"]
                st.dataframe(mat.style.format("{:.4f}"))
                st.markdown("**Matrice normalisée** (a_ij / s_j) ; le poids est la moyenne de chaque ligne")
                norm = pd.DataFrame(wres["normalized"], index=crits, columns=crits)
                norm["Poids w_i"] = wres["weights"]
                st.dataframe(norm.style.format("{:.4f}"))
                st.markdown(
                    f"λmax = Σ s_j·w_j = {fmt(wres['lambda_max'])} ; CI = (λmax − n)/(n − 1) = {fmt(wres['CI'])} ; "
                    f"CR = CI/RI = {fmt(wres['CR'])}.  \n"
                    f"Pour comparaison, le vecteur propre exact donne : "
                    + ", ".join(f"{c} {fmt(v)}" for c, v in zip(crits, wres["exact_weights"]))
                    + f" (λmax = {fmt(wres['exact_lambda_max'])})."
                )

        elif w_method == "BWM":
            st.metric("ξ*", fmt(wres["xi"]), help="Écart maximal aux jugements : plus il est proche de 0, plus les jugements sont cohérents.")
            if wres["xi"] < 0.1:
                st.success(f"ξ* = {fmt(wres['xi'])} est proche de 0 : les jugements sont cohérents.", icon=":material/check_circle:")
            else:
                st.warning(
                    f"ξ* = {fmt(wres['xi'])} : les jugements s'écartent sensiblement d'un système parfaitement cohérent. "
                    "Vérifiez les notes BO/OW (voir le tableau de cohérence ci-dessous).",
                    icon=":material/warning:",
                )
            with st.expander("Étapes de calcul BWM", icon=":material/calculate:"):
                st.markdown(
                    "Programme linéaire du cours : min ξ s.c. |w_B − a_Bj·w_j| ≤ ξ, |w_j − a_jW·w_W| ≤ ξ, "
                    "Σ w_j = 1, w_j ≥ 0. Variables x = (w_1, …, w_n, ξ), contraintes sous la forme A·x ≤ 0 :"
                )
                labels = []
                for c in crits:
                    labels += [f"w_B − a_B·w[{c}] ≤ ξ", f"−w_B + a_B·w[{c}] ≤ ξ",
                               f"w[{c}] − a_W·w_W ≤ ξ", f"−w[{c}] + a_W·w_W ≤ ξ"]
                st.dataframe(pd.DataFrame(wres["A_ub"], index=labels, columns=[*crits, "ξ"]).style.format("{:g}"))
                bo = [1 if c == ss.bwm_store["best"] else ss.bwm_store["bo"].get(c, 1) for c in crits]
                ow = [1 if c == ss.bwm_store["worst"] else ss.bwm_store["ow"].get(c, 1) for c in crits]
                st.markdown(f"**Cohérence des jugements** : idéalement a_Bj × a_jW = a_BW = {wres['a_bw']:g} pour tout j.")
                st.dataframe(pd.DataFrame({"Critère": crits, "a_Bj": bo, "a_jW": ow, "a_Bj × a_jW": wres["products"]}),
                             hide_index=True)

        elif w_method == "Entropie":
            with st.expander("Étapes de calcul de l'entropie", icon=":material/calculate:"):
                st.markdown(
                    "1) Normalisation (critère max : x/max ; critère min : min/x) ; 2) p_ij = d_ij / Σ_i d_ij ; "
                    f"3) E_j = −k Σ p_ij ln p_ij avec k = 1/ln m = {fmt(wres['k'])} ; 4) w_j = (1 − E_j)/(n − Σ E_k)."
                )
                st.dataframe(pd.DataFrame(wres["normalized"], index=alts, columns=crits).style.format("{:.4f}"))
                st.markdown("**Probabilités p_ij**")
                st.dataframe(pd.DataFrame(wres["P"], index=alts, columns=crits).style.format("{:.4f}"))
                st.dataframe(pd.DataFrame(
                    {"Normalisation": wres["normalization"], "Entropie E_j": wres["entropy"],
                     "Diversification 1 − E_j": wres["diversification"], "Poids": wres["weights"]}, index=crits,
                ).style.format({"Entropie E_j": "{:.4f}", "Diversification 1 − E_j": "{:.4f}", "Poids": "{:.4f}"}))
                st.caption("Plus l'entropie est faible (valeurs dispersées), plus le critère discrimine et plus son poids est élevé.")

        elif w_method == "CRITIC":
            with st.expander("Étapes de calcul CRITIC", icon=":material/calculate:"):
                st.markdown("**1) Matrice normalisée min-max** (max : (x − min)/(max − min) ; min : (max − x)/(max − min))")
                st.dataframe(pd.DataFrame(wres["normalized"], index=alts, columns=crits).style.format("{:.4f}"))
                st.markdown("**2) Corrélations de Pearson ρ_jk**")
                st.dataframe(pd.DataFrame(wres["rho"], index=crits, columns=crits).style.format("{:.4f}"))
                st.markdown("**3) Indice C_j = σ_j · Σ_k (1 − |ρ_jk|)** et **4) poids w_j = C_j / Σ C_j**")
                st.dataframe(pd.DataFrame(
                    {"σ_j": wres["sigma"], "Σ(1 − |ρ_jk|)": wres["independence"], "C_j": wres["C"], "Poids": wres["weights"]},
                    index=crits,
                ).style.format("{:.4f}"))
                st.caption("Un critère dispersé et peu corrélé aux autres apporte plus d'information : son poids est plus élevé.")

# ---------------------------------------------------------------------------
# 5. Classement
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.subheader("5. Classement des alternatives")
    st.caption(f"Méthode choisie dans le panneau latéral : **{r_method}**.")
    st.caption(
        "WSM = *Weighted Sum Method*, « méthode des sommes pondérées » du cours (diapo 58). "
        "L'abréviation « SWM » de la consigne désigne cette même méthode : le nom du cours, WSM, est retenu."
    )
    if r_method == "AHP":
        st.markdown(
            "Pour chaque critère, comparez les alternatives deux à deux sur l'échelle de Saaty : placez le curseur du "
            "côté de l'alternative **préférée** selon ce critère (pour un critère à minimiser comme un coût, "
            "l'alternative la moins chère est la préférée)."
        )
        st.button(
            "Préremplir depuis la matrice de décision", icon=":material/auto_fix_high:",
            on_click=prefill_ahp_alt_from_matrix, disabled=X is None,
            help="Rapport des valeurs de la matrice (inversé pour un critère à minimiser), arrondi à l'échelle 1 à 9.",
        )
        alt_pairs = [(i, j) for i in range(m) for j in range(i + 1, m)]
        if all(abs(ss.ahp_alt_store.get((c, alts[i], alts[j]), 1.0) - 1) < 1e-9 for c in crits for i, j in alt_pairs):
            st.info(
                "Toutes les comparaisons sont à « égale importance » : les alternatives seront ex æquo. "
                "Ajustez les curseurs ou utilisez le préremplissage.",
                icon=":material/info:",
            )
        for c, tab in zip(crits, st.tabs(crits)):
            with tab:
                cols = st.columns(2)
                for k, (i, j) in enumerate(alt_pairs):
                    ai, aj = alts[i], alts[j]
                    key = f"_ahpa::{c}::{ai}::{aj}"
                    if key not in ss:
                        v = ss.ahp_alt_store.get((c, ai, aj))
                        if v is None and (c, aj, ai) in ss.ahp_alt_store:
                            v = 1 / ss.ahp_alt_store[(c, aj, ai)]
                        ss[key] = snap_saaty(v or 1.0)
                    with cols[k % 2]:
                        ss.ahp_alt_store[(c, ai, aj)] = st.select_slider(
                            f"{ai} ↔ {aj}", options=mcdm.SAATY_VALUES, key=key,
                            format_func=lambda v, ai=ai, aj=aj: saaty_label(v, ai, aj),
                        )
                loc = mcdm.ahp_weights(ahp_alt_matrix(c))
                c1, c2 = st.columns([3, 2])
                c1.dataframe(pd.DataFrame({"Alternative": alts, "Priorité locale": loc["weights"]}), hide_index=True,
                             column_config={"Priorité locale": st.column_config.ProgressColumn(
                                 format="%.4f", min_value=0.0, max_value=1.0)})
                if m <= 2:
                    c2.info("2 alternatives : matrice toujours cohérente.", icon=":material/info:")
                elif loc["consistent"]:
                    c2.success(f"CR = {fmt(loc['CR'])} < 0,1 : cohérent.", icon=":material/check_circle:")
                else:
                    c2.error(f"CR = {fmt(loc['CR'])} ≥ 0,1 : jugements incohérents, à réviser.", icon=":material/error:")
        st.caption(f"{len(alt_pairs)} comparaisons par critère, soit {len(alt_pairs) * n} au total.")

    if weights is None:
        st.info("Obtenez d'abord des poids valides à l'étape 4.", icon=":material/info:")
        st.stop()

    res = rank(r_method, weights)
    show_warnings(res)
    if r_method == "AHP" and res["inconsistent"]:
        st.error(
            "Comparaisons incohérentes (CR ≥ 0,1) pour : " + ", ".join(res["inconsistent"])
            + ". Selon le cours, révisez-les avant de conclure.",
            icon=":material/error:",
        )
        if not st.checkbox("Classer malgré l'incohérence", key="force_ahp_alt"):
            st.stop()
    order = np.argsort(res["ranks"], kind="stable")
    score_label = {"WSM": "Score Q_i", "TOPSIS": "Coefficient RC_i", "AHP": "Priorité finale"}[r_method]
    table = pd.DataFrame({
        "Rang": res["ranks"][order],
        "Alternative": [alts[i] for i in order],
        score_label: res["scores"][order],
        "Ex æquo": np.where(res["tied"][order], "oui", ""),
    })
    winners = [alts[i] for i in range(m) if res["ranks"][i] == 1]
    runner = [res["scores"][i] for i in order if res["ranks"][i] > 1]

    c1, c2, c3 = st.columns(3)
    c1.metric("Meilleure alternative", " / ".join(winners))
    c2.metric(score_label, fmt(res["scores"][order[0]]))
    if runner:
        c3.metric("Écart avec la suivante", fmt(res["scores"][order[0]] - runner[0]))

    c1, c2 = st.columns([2, 3])
    c1.dataframe(table, hide_index=True, column_config={score_label: st.column_config.NumberColumn(format="%.4f")})
    c2.bar_chart(table, x="Alternative", y=score_label, horizontal=True, sort=f"-{score_label}",
                 height=max(160, 40 * m))

    # Explication en français
    top_c = crits[int(np.argmax(weights))]
    if len(winners) > 1:
        verdict = f"**{' et '.join(winners)}** sont ex æquo en tête ; départagez-les avec une autre méthode ou une analyse de sensibilité."
    else:
        verdict = f"**{winners[0]}** arrive en tête."
    if r_method == "WSM":
        how = ("WSM normalise chaque critère (x/max pour un critère à maximiser, min/x pour un critère à minimiser), "
               "puis additionne les valeurs pondérées : Q_i = Σ w_j·r_ij. Un bon résultat sur un critère peut compenser "
               "un mauvais résultat sur un autre.")
    elif r_method == "AHP":
        how = ("AHP combine les priorités locales de chaque alternative (issues de vos comparaisons par paires, "
               "critère par critère) avec les poids des critères : priorité finale = Σ w_j·w_ij.")
    else:
        how = ("TOPSIS mesure la distance de chaque alternative à la solution idéale positive (meilleures valeurs pondérées) "
               "et à la solution idéale négative. RC_i = S⁻/(S⁺ + S⁻) vaut 1 pour une alternative confondue avec l'idéal "
               "et 0 pour l'anti-idéal.")
    st.markdown(
        f"{verdict} Les poids proviennent de la méthode **{w_method}** ; le critère le plus lourd est "
        f"**{top_c}** ({fmt(weights.max(), 3)}). {how}"
    )
    if runner and res["scores"][order[0]] - runner[0] < 0.02:
        st.warning("L'écart entre les deux premières alternatives est faible : le classement peut changer avec d'autres poids.",
                   icon=":material/warning:")

    with st.expander(f"Étapes de calcul {r_method}", icon=":material/calculate:"):
        if r_method == "WSM":
            st.markdown("**Matrice normalisée r_ij**")
            st.dataframe(pd.DataFrame(res["normalized"], index=alts, columns=crits).style.format("{:.4f}"))
            st.caption("Normalisation par colonne : " + ", ".join(f"{c} : {k}" for c, k in zip(crits, res["normalization"])))
            st.markdown("**Valeurs pondérées w_j·r_ij** et score Q_i")
            wtd = pd.DataFrame(res["weighted"], index=alts, columns=crits)
            wtd["Q_i"] = res["scores"]
            st.dataframe(wtd.style.format("{:.4f}"))
        elif r_method == "AHP":
            st.markdown("**Priorités locales w_ij** (une colonne par critère) et **priorités finales** Σ w_j·w_ij")
            loc_df = pd.DataFrame(res["local"], index=alts, columns=crits)
            loc_df["Priorité finale"] = res["scores"]
            loc_df.loc["Poids des critères w_j"] = [*weights, np.nan]
            st.dataframe(loc_df.style.format("{:.4f}", na_rep=""))
            st.dataframe(pd.DataFrame(
                {"λmax": [r["lambda_max"] for r in res["local_results"]],
                 "CI": [r["CI"] for r in res["local_results"]],
                 "CR": [r["CR"] for r in res["local_results"]]}, index=crits,
            ).style.format("{:.4f}"))
            for c in crits:
                st.markdown(f"**Matrice de comparaison des alternatives — {c}**")
                st.dataframe(pd.DataFrame(ahp_alt_matrix(c), index=alts, columns=alts).style.format("{:.4f}"))
        else:
            st.markdown("**Étape 1 : normalisation vectorielle** r_ij = x_ij / √(Σ_i x_ij²)")
            st.dataframe(pd.DataFrame(res["normalized"], index=alts, columns=crits).style.format("{:.4f}"))
            st.markdown("**Étapes 2-3 : matrice pondérée** v_ij = w_j·r_ij **et solutions idéales**")
            v = pd.DataFrame(res["weighted"], index=alts, columns=crits)
            v.loc["I⁺ (idéal positif)"] = res["ideal_pos"]
            v.loc["I⁻ (idéal négatif)"] = res["ideal_neg"]
            st.dataframe(v.style.format("{:.4f}"))
            st.markdown("**Étapes 4-5 : séparations et coefficient de proximité**")
            st.dataframe(pd.DataFrame(
                {"S⁺": res["s_pos"], "S⁻": res["s_neg"], "RC_i": res["scores"]}, index=alts
            ).style.format("{:.4f}"))

    export = table.copy()
    export.insert(0, "Problème", ss.problem)
    export["Pondération"] = w_method
    export["Classement"] = r_method
    for c, wj in zip(crits, weights):
        export[f"Poids {c}"] = wj
    csv = export.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
    st.download_button(
        "Exporter le classement (CSV)", csv, file_name=f"classement_{w_method}_{r_method}_{date.today()}.csv",
        mime="text/csv", icon=":material/download:", on_click="ignore",
    )

# ---------------------------------------------------------------------------
# 6. Comparaison des 8 combinaisons
# ---------------------------------------------------------------------------

with st.container(border=True):
    st.subheader("6. Comparer les combinaisons")
    st.caption("Le cours recommande d'appliquer plusieurs méthodes et de comparer les classements obtenus.")
    ranks, notes = {}, []
    for wm in WEIGHT_METHODS:
        w_, r_ = compute_weights(wm)
        if w_ is None:
            notes.append(f"{wm} : {r_ if isinstance(r_, str) else 'jugements incohérents (CR ≥ 0,1)'}")
            continue
        for rm in RANK_METHODS:
            if rm != "AHP" and X is None:
                continue
            rr = rank(rm, w_)
            if rm == "AHP" and rr["inconsistent"] and not ss.get("force_ahp_alt", False):
                continue
            ranks[f"{wm} + {rm}"] = rr["ranks"]
    if X is None:
        notes.append("WSM, TOPSIS, entropie et CRITIC : matrice de décision incomplète")
    if notes:
        st.caption("Combinaisons non calculées — " + " ; ".join(notes))
    if ranks:
        comp = pd.DataFrame(ranks, index=alts)
        comp.index.name = "Alternative"
        st.dataframe(comp)
        current = f"{w_method} + {r_method}"
        if current in ranks and len(ranks) > 1:
            rho = pd.DataFrame(
                {"Corrélation de Spearman avec " + current: [mcdm.spearman(ranks[current], ranks[k]) for k in ranks]},
                index=list(ranks),
            )
            st.dataframe(rho.style.format("{:.3f}", na_rep="—"))
            st.caption("1 : classements identiques ; 0 : aucun lien ; −1 : classements inversés.")
