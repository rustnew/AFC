# 🧮 Analyse Factorielle des Correspondances (AFC) — Projet Jupyter

## 📘 Introduction

L’**Analyse Factorielle des Correspondances (AFC)** est une méthode statistique utilisée pour **analyser un tableau de contingence** (tableau de comptages) et **représenter les relations entre les lignes et les colonnes** dans un espace de faible dimension.
Elle est particulièrement pertinente pour **l’étude de données qualitatives**, lorsque l’on souhaite identifier des **profils similaires** ou des **corrélations entre modalités**.

Dans ce projet, nous implémentons **une AFC complète en Python**, étape par étape, **sans bibliothèque spécialisée** (comme `prince` ou `factoMineR`), afin de **maîtriser les fondements mathématiques et algorithmiques** de la méthode.

Les données sont **générées artificiellement** pour simuler un tableau de contingence réaliste :

```python
import numpy as np
import pandas as pd

np.random.seed(42)
n_lignes, n_colonnes = 25, 5
donnees_afc = np.random.poisson(lam=10, size=(n_lignes, n_colonnes))
lignes = [f"Individu_{i+1}" for i in range(n_lignes)]
colonnes = [f"Variable_{chr(65+j)}" for j in range(n_colonnes)]
df_afc = pd.DataFrame(donnees_afc, index=lignes, columns=colonnes)
```

---

## 🎯 Objectif

L’objectif est de **comprendre et implémenter chaque étape de l’AFC** :

1. Préparer les données et calculer les fréquences ;
2. Centrer et pondérer les données pour obtenir la matrice du khi-deux ;
3. Réaliser une décomposition en valeurs singulières (SVD) ;
4. Extraire les **valeurs propres**, les **coordonnées factorielles**, et les **contributions** ;
5. Visualiser les résultats sur les deux premiers axes factoriels.

---

## ⚙️ Étape 1 — Construction du tableau et fréquences

### But :

Transformer le tableau brut ( N ) (effectifs) en un tableau de **fréquences relatives** ( P ), puis calculer les **profils marginaux** des lignes et des colonnes.

### Formules :

[
P = \frac{N}{n}
]
[
r_i = \sum_j P_{ij}, \quad c_j = \sum_i P_{ij}
]

### Rôle :

* ( P ) : pondère les effectifs pour supprimer l’effet de la taille totale.
* ( r ) et ( c ) : représentent les **poids** (ou masses) des lignes et colonnes, c’est-à-dire leur importance dans le total.

```python
N = df_afc.values
n_total = N.sum()
P = N / n_total
r = P.sum(axis=1).reshape(-1, 1)
c = P.sum(axis=0).reshape(1, -1)
```

---

## ⚙️ Étape 2 — Centrage et pondération du tableau

### But :

Extraire la **structure d’association** entre les lignes et les colonnes en éliminant l’effet des marges.

### Formule :

[
S = D_r^{-1/2} (P - r c) D_c^{-1/2}
]
où :

* ( D_r ) et ( D_c ) sont les matrices diagonales contenant respectivement ( r_i ) et ( c_j ) ;
* ( P - rc ) mesure les écarts entre la fréquence observée et la fréquence théorique sous indépendance.

### Rôle :

Cette étape recentre les données autour de l’hypothèse d’indépendance et **met toutes les lignes et colonnes sur un pied d’égalité**.

```python
Dr_inv_sqrt = np.diag(1 / np.sqrt(r.flatten()))
Dc_inv_sqrt = np.diag(1 / np.sqrt(c.flatten()))
S = Dr_inv_sqrt @ (P - r @ c) @ Dc_inv_sqrt
```

---

## ⚙️ Étape 3 — Décomposition en valeurs singulières (SVD)

### But :

Extraire les **axes factoriels principaux** qui expliquent la variance (inertie) du nuage de points.

### Formule :

[
S = U \Sigma V^T
]
où :

* ( \Sigma ) : matrice diagonale des **valeurs singulières** ;
* ( \lambda_i = \sigma_i^2 ) : **valeurs propres** (inerties).

### Rôle :

La SVD permet de **projeter les lignes et colonnes dans un même espace**.
Chaque axe factoriel correspond à une **dimension latente d’association** entre lignes et colonnes.

```python
U, singular_values, VT = np.linalg.svd(S, full_matrices=False)
eigenvalues = singular_values**2
inertie = 100 * eigenvalues / eigenvalues.sum()
```

---

## ⚙️ Étape 4 — Coordonnées factorielles

### Formules :

[
F = D_r^{-1/2} U \Sigma
]
[
G = D_c^{-1/2} V \Sigma
]

### Rôle :

* ( F ) : coordonnées factorielles des lignes sur les axes principaux.
* ( G ) : coordonnées factorielles des colonnes sur les mêmes axes.

Elles permettent de **représenter visuellement** la proximité entre lignes et colonnes.

```python
V = VT.T
F = Dr_inv_sqrt @ U @ np.diag(singular_values)
G = Dc_inv_sqrt @ V @ np.diag(singular_values)
```

---

## ⚙️ Étape 5 — Contributions et qualités de représentation

### But :

Évaluer l’importance de chaque point (ligne ou colonne) dans la construction des axes.

### Formules :

[
\text{CTR}*{ij} = \frac{r_i F*{ij}^2}{\lambda_j}
\quad ; \quad
\text{COS2}*{ij} = \frac{F*{ij}^2}{\sum_k F_{ik}^2}
]

### Rôle :

* **CTR (contribution)** : indique combien chaque ligne/colonne contribue à un axe ;
* **COS² (qualité de représentation)** : mesure la qualité du placement du point sur un axe (analogue à un ( R^2 )).

```python
CTR_rows = (r * (F**2)) / eigenvalues
CTR_rows = CTR_rows / CTR_rows.sum(axis=0)
COS2_rows = (F**2) / F.sum(axis=1, keepdims=True)**2
```

---

## ⚙️ Étape 6 — Visualisation du plan factoriel

### But :

Visualiser les relations entre individus (lignes) et variables (colonnes).

### Rôle :

* Les **points proches** traduisent des **profils similaires**.
* Les **axes principaux** concentrent l’essentiel de l’information.
* Les **lignes** et **colonnes** peuvent être représentées conjointement.

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(F_df["Axe_1"], F_df["Axe_2"], color='blue', label="Lignes")
plt.scatter(G_df["Axe_1"], G_df["Axe_2"], color='red', marker='x', label="Colonnes")

for i, txt in enumerate(df_afc.index):
    plt.annotate(txt, (F_df["Axe_1"].iloc[i], F_df["Axe_2"].iloc[i]), fontsize=8)

for j, txt in enumerate(df_afc.columns):
    plt.annotate(txt, (G_df["Axe_1"].iloc[j], G_df["Axe_2"].iloc[j]), color='red', fontsize=9)

plt.axhline(0, color='gray', linewidth=0.8)
plt.axvline(0, color='gray', linewidth=0.8)
plt.title("Plan factoriel (Axes 1 et 2) - Analyse Factorielle des Correspondances")
plt.xlabel("Axe 1")
plt.ylabel("Axe 2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
```

---

## 📊 Interprétation pratique

1. **Valeurs propres (inertie)** : mesurent la part de variance expliquée par chaque axe.
   → Les deux premiers axes concentrent souvent 60–80 % de l’information.
2. **Coordonnées factorielles** : permettent d’identifier les profils proches ou opposés.
3. **CTR et COS²** : aident à savoir **quelles lignes ou colonnes** sont les plus importantes sur un axe.
4. **Graphique factoriel** : synthétise visuellement les associations entre modalités.

---

## 🧠 Bilan et pertinence de l’AFC

L’AFC est une méthode :

* **descriptive** (aucune hypothèse préalable) ;
* **exploratoire** (fait émerger des structures cachées) ;
* **visuelle** (les cartes factoriels offrent une lecture intuitive des corrélations).

Dans la pratique :

* Elle est utilisée en **marketing**, **sociologie**, **analyse textuelle**, ou **études d’opinion**.
* Elle permet de **résumer un grand tableau de données qualitatives** en quelques axes interprétables.

---

## 🧩 Références théoriques

* Benzécri, J.-P. (1973). *L’Analyse des Données — Tome 2 : L’Analyse des Correspondances*. Dunod.
* Greenacre, M. (2017). *Correspondence Analysis in Practice*. Chapman & Hall/CRC.
* Saporta, G. (2006). *Probabilités, analyse des données et statistique*. Technip.

---

## 🏁 Conclusion

Ce travail illustre :

* La **traduction directe des formules mathématiques** en code Python ;
* La **démarche complète d’une AFC**, du tableau brut à l’interprétation graphique ;
* La **valeur pédagogique** de reconstruire soi-même l’algorithme sans dépendre de bibliothèques toutes faites.

L’implémentation manuelle démontre que l’AFC repose sur :

* Une **logique matricielle élégante** (centrage, normalisation, SVD) ;
* Une **visualisation intuitive** qui relie théorie et interprétation pratique.
