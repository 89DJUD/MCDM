"""Vérification des calculs sur les exercices du cours (lancer : python -m pytest)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mcdm  # noqa: E402


def test_ahp_voiture():
    # Diapo 33 : coût / confort / sécurité.
    r = mcdm.ahp_weights([[1, 7, 3], [1 / 7, 1, 1 / 3], [1 / 3, 3, 1]])
    # Sommes des colonnes : 1,4762 ; 11 ; 4,3333.
    np.testing.assert_allclose(r["col_sums"], [1 + 1 / 7 + 1 / 3, 11, 13 / 3])
    # Poids = moyennes des lignes de la matrice normalisée.
    expected = np.mean([[1 / (31 / 21), 7 / 11, 3 / (13 / 3)],
                        [(1 / 7) / (31 / 21), 1 / 11, (1 / 3) / (13 / 3)],
                        [(1 / 3) / (31 / 21), 3 / 11, 1 / (13 / 3)]], axis=1)
    np.testing.assert_allclose(r["weights"], expected)
    np.testing.assert_allclose(r["weights"], [0.6687, 0.0882, 0.2431], atol=1e-4)
    assert r["CR"] < 0.1 and r["consistent"]


def test_ahp_voiture_priorites_finales():
    w = mcdm.ahp_weights([[1, 7, 3], [1 / 7, 1, 1 / 3], [1 / 3, 3, 1]])["weights"]
    local = np.column_stack([
        mcdm.ahp_weights([[1, 7], [1 / 7, 1]])["weights"],
        mcdm.ahp_weights([[1, 0.2], [5, 1]])["weights"],
        mcdm.ahp_weights([[1, 1 / 9], [9, 1]])["weights"],
    ])
    final = local @ w
    np.testing.assert_allclose(local[:, 0], [0.875, 0.125])
    assert final[0] > final[1]  # la voiture 1 l'emporte (≈ 0,624 contre 0,376)


def test_ahp_incoherent():
    r = mcdm.ahp_weights(mcdm.ahp_matrix_from_judgments(3, {(0, 1): 9, (1, 2): 9, (0, 2): 1 / 9}))
    assert r["CR"] > 0.1 and not r["consistent"]


def test_bwm_fournisseur():
    # Diapos 38-44 : B = qualité, W = RSE.
    r = mcdm.bwm_weights(0, 3, [1, 3, 4, 8], [8, 5, 3, 1])
    assert r["xi"] == pytest.approx(0.0838, abs=1e-4)  # le cours annonce ξ* ≈ 0,08
    np.testing.assert_allclose(r["weights"], [0.5629, 0.2156, 0.1617, 0.0599], atol=1e-4)
    assert r["weights"].sum() == pytest.approx(1)
    # Toutes les contraintes |w_B − a_Bj w_j| ≤ ξ et |w_j − a_jW w_W| ≤ ξ sont satisfaites.
    assert (r["A_ub"] @ np.r_[r["weights"], r["xi"]] <= 1e-9).all()


def test_bwm_meme_critere():
    with pytest.raises(ValueError):
        mcdm.bwm_weights(1, 1, [1, 2], [2, 1])


def test_critic_machines():
    # Diapo 53.
    X = [[30, 0.1, 1, 20], [100, 0.7, 1, 40], [50, 1, 2, 10], [300, 2, 3, 35]]
    r = mcdm.critic_weights(X, [False, False, True, False])
    np.testing.assert_allclose(r["normalized"][:, 0], [1, 200 / 270, 250 / 270, 0])
    np.testing.assert_allclose(r["weights"], [0.1436, 0.1595, 0.2639, 0.4331], atol=1e-4)


def test_wsm_machines():
    # Diapo 60.
    X = [[0.035, 847, 0.335, 1.760, 0.590],
         [0.027, 834, 0.335, 1.680, 0.665],
         [0.037, 808, 0.590, 2.400, 0.500],
         [0.028, 821, 0.500, 1.590, 0.410]]
    w = [0.331, 0.181, 0.369, 0.072, 0.047]
    r = mcdm.wsm(X, [False, False, True, False, True], w)
    # Calcul manuel de Q_1.
    q1 = (0.331 * 0.027 / 0.035 + 0.181 * 808 / 847 + 0.369 * 0.335 / 0.590
          + 0.072 * 1.590 / 1.760 + 0.047 * 0.590 / 0.665)
    assert r["scores"][0] == pytest.approx(q1)
    np.testing.assert_allclose(r["scores"], [0.7443, 0.8310, 0.8746, 0.9110], atol=1e-4)
    assert list(r["ranks"]) == [4, 3, 2, 1]


def test_topsis_voitures():
    # Diapo 67.
    X = [[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6]]
    r = mcdm.topsis(X, [True, True, True, False], [0.2, 0.1, 0.4, 0.3])
    assert r["normalized"][0, 0] == pytest.approx(7 / np.sqrt(49 + 64 + 81 + 36))
    np.testing.assert_allclose(r["scores"], [0.4545, 0.5678, 0.3703, 0.5527], atol=1e-4)
    assert list(r["ranks"]) == [3, 1, 4, 2]  # M2 est la meilleure marque


def test_entropie_critere_constant():
    X = [[1, 5], [2, 5], [3, 5]]
    r = mcdm.entropy_weights(X, [True, True])
    assert r["weights"][1] == pytest.approx(0)
    assert r["weights"].sum() == pytest.approx(1)


def test_entropie_tout_constant():
    r = mcdm.entropy_weights([[2, 5], [2, 5]], [True, False])
    np.testing.assert_allclose(r["weights"], [0.5, 0.5])


def test_critic_constant_et_zero():
    r = mcdm.critic_weights([[1, 0, 4], [2, 0, 1], [3, 0, 2]], [True, False, True])
    assert r["weights"][1] == pytest.approx(0)
    assert np.isfinite(r["weights"]).all()


def test_wsm_valeurs_nulles():
    r = mcdm.wsm([[0, 1], [2, 3], [4, 0]], [True, False], [0.5, 0.5])
    assert np.isfinite(r["scores"]).all() and r["warnings"]


def test_ex_aequo():
    r = mcdm.topsis([[1, 1], [1, 1], [2, 0]], [True, False], [0.5, 0.5])
    assert r["ranks"][0] == r["ranks"][1] and r["tied"][0]
    ranks, tied = mcdm.rank_scores([0.5, 0.7, 0.7, 0.1])
    assert list(ranks) == [3, 1, 1, 4] and list(tied) == [False, True, True, False]


def test_valeurs_manquantes():
    with pytest.raises(ValueError):
        mcdm.wsm([[1, np.nan], [2, 3]], [True, True], [0.5, 0.5])
