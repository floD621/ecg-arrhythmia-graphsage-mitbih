# Classification d'arythmies à partir de graphes de battements ECG

## Résumé

Ce travail présente une approche de classification binaire (normal vs arythmie) sur la base MIT-BIH, en représentant chaque segment ECG comme un graphe de battements. Les nœuds portent des descripteurs morpho-temporels, les arêtes encodent principalement la proximité temporelle, et la décision est produite par un réseau de neurones sur graphes de type GraphSAGE.

## 1. Introduction

La détection automatique des arythmies sur ECG est un problème clinique important. Les approches purement séquentielles (CNN/RNN/Transformers) exploitent bien la morphologie locale, mais modélisent moins explicitement les relations entre battements successifs. La représentation en graphe permet de capturer ces dépendances inter-battements, utiles pour les motifs rythmiques.

## 2. Données et annotations

- **Dataset**: MIT-BIH Arrhythmia Database (PhysioNet).
- **Contenu**: 48 enregistrements ECG annotés battement par battement.
- **Fréquence d'échantillonnage**: 360 Hz.
- **Préparation**:
  - lecture des signaux et annotations via `wfdb`,
  - extraction des battements autour des pics R annotés,
  - segmentation en fenêtres de taille fixe (`window_size`, `stride`).

## 3. Méthodologie

### 3.1 Construction du graphe

Pour chaque segment:

- **Nœuds**: battements ECG extraits.
- **Features de nœud** (implémentation actuelle): descripteurs compacts (RR, amplitude, statistiques locales).
- **Arêtes**: voisinage temporel (liaison au(x) battement(s) adjacent(s)).
- **Label de graphe**: binaire (`normal`/`arrhythmia`) défini à partir des annotations du segment.

### 3.2 Modèles

- **GraphSAGE**: agrégation de voisinage robuste et simple.
- **Tête de classification**: couche linéaire sur l'embedding global du graphe.

### 3.3 Entraînement

- Fonction de coût: entropie croisée.
- Optimiseur: Adam.
- Métriques suivies: `accuracy` et `f1_macro`.
- Stratégie de checkpoint: sauvegarde du meilleur modèle selon `f1_macro` validation dans `weights/gnn_best.pth`.

## 4. Résultats

### 4.1 Résultats observés (préliminaires)

Résultats mesurés dans le notebook d'évaluation (`notebooks/3_evaluate_gnn.ipynb`), après chargement du meilleur checkpoint `weights/gnn_best.pth`:

- Accuracy: **0.975824**
- F1-macro: **0.975254**
- Taille du test: **4674 graphes**
- Matrice de confusion: **[[2635, 47], [66, 1926]]**
- Meilleur checkpoint: **epoch 97**, `best_metric = 0.9752543093138715`

Interprétation rapide:

- Les performances sont élevées et relativement équilibrées entre classes.
- Le nombre de faux négatifs arythmie (66) reste non nul et doit être suivi en priorité clinique.

### 4.2 Résultats à reporter (run 100 epochs)

Tableau des résultats consolidés:

| Modèle | Epochs | Accuracy | F1-macro | Checkpoint epoch | Commentaire |
| --- | --- | --- | --- | --- | --- |
| GraphSAGE | 100 | 0.975824 | 0.975254 | 97 | Très bon niveau global; vigilance sur faux négatifs arythmie |

## 5. Visualisations

Le notebook `notebooks/3_evaluate_gnn.ipynb` fournit:

- rapport de classification,
- matrice de confusion,
- projection t-SNE des embeddings de graphes,
- visualisation d'un graphe de battements.

Ces vues permettent de diagnostiquer:

- séparation inter-classes dans l'espace latent,
- confusion dominante (normal -> arythmie ou inverse),
- cohérence structurelle des graphes construits.

## 6. Regard critique

### 6.1 Forces

- Pipeline cohérent de bout en bout (données réelles -> graphe -> entraînement -> checkpoint -> visualisation).
- Architecture cohérente et stable (GraphSAGE).
- Suivi d'une métrique plus pertinente que l'accuracy seule (`f1_macro`).

### 6.2 Limites méthodologiques

- **Split potentiel par segments seulement**: risque de fuite d'information si des segments proches d'un même enregistrement sont dans train et test.
- **Déséquilibre de classes**: pénalise la détection d'arythmie.
- **Features de nœud simples**: peuvent limiter la richesse morphologique exploitée.
- **Évaluation incomplète**: absence actuelle d'AUC-PR, sensibilité/spécificité par classe, intervalles de confiance.

### 6.3 Menaces à la validité

- Variabilité inter-patient insuffisamment contrôlée sans split patient-wise strict.
- Résultats dépendants des hyperparamètres de segmentation (`window_size`, `stride`).
- Reproductibilité à renforcer (journal complet des seeds, versions et configs run).

## 7. Améliorations recommandées

1. Mettre en place un split patient-wise strict (ou leave-record-out).
2. Ajouter des pondérations de classes ou une focal loss.
3. Étendre les métriques (AUC-PR, recall classe arythmie, balanced accuracy).
4. Tester des features morphologiques plus riches (embeddings appris par encodeur battement).
5. Faire une étude d'ablation: sans arêtes temporelles, variation de `k` voisins, variation de profondeur/largeur de GraphSAGE à budget constant.

## 8. Conclusion

Les résultats obtenus avec GraphSAGE sur le split évalué sont solides: accuracy = 0.975824, F1-macro = 0.975254, avec une matrice de confusion [[2635, 47], [66, 1926]]. Ces chiffres indiquent une bonne séparation globale des classes et une performance plus équilibrée que ce qu'on observe habituellement dans des contextes très déséquilibrés.

D'un point de vue applicatif, la vigilance principale reste les faux négatifs sur la classe arythmie (66 cas dans ce run), car ce sont les erreurs les plus critiques cliniquement. En conséquence, ce travail valide la faisabilité technique de l'approche GraphSAGE sur MIT-BIH, mais ne suffit pas encore à revendiquer une robustesse clinique: il faut confirmer ces performances avec une validation patient-wise stricte, plusieurs seeds, et des métriques orientées dépistage (recall arythmie, AUC-PR, calibration).

## Annexe: checklist pour version article

- Décrire précisément le protocole de split.
- Rapporter moyenne ± écart-type sur plusieurs seeds.
- Ajouter comparaison entre variantes GraphSAGE (profondeur, hidden size, ablations).
- Discuter le coût calcul/mémoire et temps d'inférence.
- Inclure les erreurs types (cas d'arythmies manquées) avec exemples visuels.
