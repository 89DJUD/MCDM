# Aide à la décision multicritère (Streamlit)

Application d'aide à la décision multicritère (MCDM) construite sur les formules du cours
« Aide à la décision » (Prof. Lamrani Alaoui, EMI).

- **Pondération subjective** : AHP (méthode approximative, λmax, CI, RI, CR) et BWM (programme linéaire, ξ*).
- **Pondération objective** : entropie et CRITIC, avec toutes les étapes intermédiaires.
- **Classement** : WSM (somme pondérée), TOPSIS et AHP (priorités locales par comparaisons des alternatives, puis priorités finales).

Chacune des 4 méthodes de pondération se combine avec chacune des 3 méthodes de classement,
soit 12 combinaisons. Une section compare les classements, avec la corrélation de Spearman.

## « SWM » ou « WSM » ?

Le cours (diapo 58) nomme la méthode *« méthode des sommes pondérées (weighted sum method ou WSM) »* :
Q_i = Σ_j w_j·r_ij. « SWM » dans la consigne est une inversion de lettres. L'application emploie
donc **WSM** et le signale dans l'interface.

## Structure

| Fichier | Rôle |
|---|---|
| `app.py` | Interface Streamlit |
| `mcdm.py` | Calculs (AHP, BWM, entropie, CRITIC, WSM, TOPSIS, rangs, Spearman) |
| `examples.py` | Exemples préremplis tirés des exercices du cours |
| `tests/test_mcdm.py` | Vérification chiffrée sur les exercices du cours |
| `requirements.txt` | Dépendances |

## Lancement local

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

Tests (facultatif) :

```bash
pip install pytest
python -m pytest
```

## Déploiement sur Streamlit Community Cloud

1. Poussez le dossier dans un dépôt GitHub (`app.py`, `mcdm.py`, `examples.py` et `requirements.txt` à la racine).
2. Sur https://share.streamlit.io, cliquez sur **Create app**, choisissez le dépôt, la branche et le fichier principal `app.py`.
3. Dans *Advanced settings*, choisissez Python 3.12 ou plus récent, puis **Deploy**.
   Les dépendances de `requirements.txt` sont installées automatiquement.

## Utilisation

1. Nommez le problème, ou chargez un exemple.
2. Modifiez les critères (sens Max ou Min) et les alternatives. Pour ajouter une ligne, utilisez le bas du tableau ;
   pour en supprimer une, sélectionnez-la puis appuyez sur Suppr.
3. Remplissez la matrice de décision.
4. Dans le panneau latéral, choisissez la méthode de pondération et la méthode de classement ;
   saisissez les jugements (AHP, BWM) si nécessaire.
5. Consultez le classement WSM, TOPSIS ou AHP : scores, classement complet, meilleure alternative, explication et export CSV
   (séparateur `;`, décimale `,`, lisible directement par Excel en français).

## Choix de calcul et cas particuliers

- **AHP** : poids obtenus par normalisation des colonnes puis moyenne des lignes, λmax = Σ s_j·w_j,
  CI = (λmax − n)/(n − 1), CR = CI/RI (table RI du cours). Si CR ≥ 0,1, le classement est bloqué et les paires les plus
  incohérentes sont signalées. Une case permet toutefois de forcer l'utilisation des poids. Le vecteur propre exact est
  affiché à titre de comparaison.
- **BWM** : modèle linéaire du cours, min ξ s.c. |w_B − a_Bj·w_j| ≤ ξ et |w_j − a_jW·w_W| ≤ ξ, résolu par
  `scipy.optimize.linprog`. Les contrôles du cours (a_BB = a_WW = 1, note BO maximale sur W) sont vérifiés, ainsi que la
  cohérence a_Bj × a_jW = a_BW.
- **Entropie** : la matrice est d'abord normalisée par les ratios du cours (x/max ou min/x), avec la convention 0·ln 0 = 0.
  Un critère constant a une entropie de 1 et donc un poids nul.
- **CRITIC** : normalisation min-max, écart type avec m − 1, corrélation de Pearson. Pour un critère constant, les
  corrélations sont fixées à 0 et le poids est nul. Avec moins de 3 alternatives, un avertissement s'affiche.
- **Classement AHP** (diapos 31-32) : pour chaque critère, matrice de comparaison des alternatives (échelle 1 à 9),
  priorités locales et CR par critère, puis priorité finale Σ w_j·w_ij. Avec une pondération AHP ou BWM, la matrice
  de décision est facultative. Un bouton préremplit les comparaisons à partir des rapports de la matrice.
- **WSM** : r = x/max (critère à maximiser), r = min/x (critère à minimiser). Si une colonne contient une valeur nulle ou
  négative, elle est normalisée par min-max et un avertissement s'affiche (pas de division par zéro).
- **TOPSIS** : normalisation vectorielle, idéaux I⁺/I⁻, RC = S⁻/(S⁺ + S⁻). Une colonne nulle donne r = 0 ; si S⁺ + S⁻ = 0,
  alors RC = 0,5.
- **Données** : valeurs manquantes, noms vides ou en double, et moins de 2 critères ou 2 alternatives sont signalés,
  et le calcul s'arrête.
- **Ex æquo** : les scores égaux (à 10⁻⁹ près) partagent le même rang (1, 2, 2, 4) et sont marqués dans le tableau.

## Vérification sur les exercices du cours

| Exercice | Résultat |
|---|---|
| AHP voiture (diapos 33-35) | w = (0,6687 ; 0,0882 ; 0,2431), λmax = 3,0108, CR = 0,0093 < 0,1. Priorités finales : voiture 1 = 0,6241, voiture 2 = 0,3759 |
| CRITIC machines de découpe (diapo 53) | w = (0,1436 ; 0,1595 ; 0,2639 ; 0,4331) |
| WSM machines (diapo 60) | Q = (0,7443 ; 0,8310 ; 0,8746 ; 0,9110) → **A4** meilleure |
| TOPSIS voitures (diapo 67) | RC = (0,4545 ; 0,5678 ; 0,3703 ; 0,5527) → **M2** meilleure |
| BWM fournisseur (diapos 38-44) | ξ* = 0,0838 (le cours annonce 0,08), w = (0,5629 ; 0,2156 ; 0,1617 ; 0,0599) |

**Remarque sur l'exemple BWM** : le cours affiche les poids (0,552 ; 0,207 ; 0,155 ; 0,086) avec ξ* = 0,08. L'optimum du
modèle linéaire décrit dans le cours est **unique** : les bornes min et max de chaque poids à ξ* coïncident. Or le vecteur
du cours ne respecte pas ses propres contraintes (par exemple |w₂ − 5·w₄| = 0,223 > 0,08). La valeur de ξ* concorde, mais
le tableau des poids du cours provient probablement d'une autre variante du modèle.
