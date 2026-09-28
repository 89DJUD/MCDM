"""Exemples préremplis, repris des exercices du cours lorsque c'est possible."""

MAX = "Max (bénéfice)"
MIN = "Min (coût)"

BLANK = "Nouveau problème (matrice vide à remplir)"

# Point de départ vierge : l'utilisateur renomme critères et alternatives et saisit sa propre matrice.
BLANK_PROBLEM = {
    "problem": "Mon problème de décision",
    "criteria": [("Critère 1", MAX), ("Critère 2", MAX), ("Critère 3", MIN)],
    "alternatives": ["Alternative 1", "Alternative 2", "Alternative 3"],
    "matrix": None,
    "ahp": {},
    "bwm": {"best": 0, "worst": 2, "bo": [1, 1, 1], "ow": [1, 1, 1]},
}

EXAMPLES = {
    "Achat d'une voiture (exercice AHP complet du cours)": {
        "problem": "Acheter une voiture",
        "criteria": [("Coût", MIN), ("Confort", MAX), ("Sécurité", MAX)],
        "alternatives": ["Voiture 1", "Voiture 2"],
        "matrix": None,  # AHP classe les alternatives par comparaisons par paires : pas de matrice chiffrée
        "ahp": {(0, 1): 7, (0, 2): 3, (1, 2): 1 / 3},
        # Comparaisons des alternatives par critère : {indice critère: {(i, j): a_ij}} (diapo 34).
        "ahp_alt": {0: {(0, 1): 7}, 1: {(0, 1): 1 / 5}, 2: {(0, 1): 1 / 9}},
        "bwm": {"best": 0, "worst": 1, "bo": [1, 7, 3], "ow": [7, 1, 3]},
        "methods": ("AHP", "AHP"),
    },
    "Choix d'une voiture (exercice TOPSIS du cours)": {
        "problem": "Choisir une voiture parmi 4 marques",
        "criteria": [("Style", MAX), ("Fiabilité", MAX), ("Économie de carburant", MAX), ("Coût", MIN)],
        "alternatives": ["M1", "M2", "M3", "M4"],
        "matrix": [[7, 9, 9, 8], [8, 7, 8, 7], [9, 6, 8, 9], [6, 7, 8, 6]],
        # Jugements AHP : importance du critère i par rapport au critère j (i < j).
        "ahp": {(0, 1): 2, (0, 2): 1 / 2, (0, 3): 1 / 2, (1, 2): 1 / 4, (1, 3): 1 / 3, (2, 3): 2},
        "bwm": {"best": 2, "worst": 1, "bo": [2, 4, 1, 2], "ow": [2, 1, 4, 3]},
    },
    "Machines de découpe (exercice CRITIC du cours)": {
        "problem": "Sélectionner une machine de découpage",
        "criteria": [
            ("Épaisseur maximale", MIN),
            ("Largeur de coupe minimale", MIN),
            ("Qualité de découpe", MAX),
            ("Coût de maintenance", MIN),
        ],
        "alternatives": ["A1", "A2", "A3", "A4"],
        "matrix": [[30, 0.1, 1, 20], [100, 0.7, 1, 40], [50, 1, 2, 10], [300, 2, 3, 35]],
        "ahp": {(0, 1): 1, (0, 2): 1 / 3, (0, 3): 1 / 2, (1, 2): 1 / 3, (1, 3): 1 / 2, (2, 3): 2},
        "bwm": {"best": 2, "worst": 0, "bo": [3, 3, 1, 2], "ow": [1, 1, 3, 2]},
    },
    "Choix d'une machine (exercice WSM/WASPAS du cours)": {
        "problem": "Choisir la meilleure machine parmi 4",
        "criteria": [("C1", MIN), ("C2", MIN), ("C3", MAX), ("C4", MIN), ("C5", MAX)],
        "alternatives": ["A1", "A2", "A3", "A4"],
        "matrix": [
            [0.035, 847, 0.335, 1.760, 0.590],
            [0.027, 834, 0.335, 1.680, 0.665],
            [0.037, 808, 0.590, 2.400, 0.500],
            [0.028, 821, 0.500, 1.590, 0.410],
        ],
        "ahp": {
            (0, 1): 2, (0, 2): 1, (0, 3): 5, (0, 4): 7,
            (1, 2): 1 / 2, (1, 3): 3, (1, 4): 4,
            (2, 3): 5, (2, 4): 7,
            (3, 4): 2,
        },
        "bwm": {"best": 2, "worst": 4, "bo": [1, 2, 1, 5, 8], "ow": [7, 4, 8, 2, 1]},
    },
    "Choix d'un fournisseur (jugements BWM du cours, données illustratives)": {
        "problem": "Sélectionner un fournisseur",
        "criteria": [("Qualité", MAX), ("Prix", MIN), ("Délai de livraison", MIN), ("Engagement RSE", MAX)],
        "alternatives": ["Fournisseur A", "Fournisseur B", "Fournisseur C"],
        "matrix": [[8, 120, 10, 6], [6, 95, 7, 8], [9, 140, 14, 5]],
        "ahp": {(0, 1): 3, (0, 2): 4, (0, 3): 8, (1, 2): 2, (1, 3): 5, (2, 3): 3},
        "bwm": {"best": 0, "worst": 3, "bo": [1, 3, 4, 8], "ow": [8, 5, 3, 1]},
    },
}
