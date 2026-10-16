# Détection d'arythmies par GNN sur battements ECG (MIT-BIH)

Prototype de classification de rythme cardiaque à partir de graphes de battements ECG.

## Objectif

- Charger les enregistrements annotés MIT-BIH.
- Segmenter les battements en fenêtres temporelles.
- Construire un graphe par segment (nœuds = battements, arêtes = voisinage temporel).
- Entraîner un modèle GNN (GraphSAGE) pour la classification binaire:
  - `0` = normal
  - `1` = arythmie

## Arborescence et rôle de chaque fichier

### Racine

- `README.md` : guide d'utilisation du projet.
- `requirements.txt` : dépendances Python principales.

### Données

- `data/mitbih/*.hea` et `data/mitbih/*.atr` : fichiers MIT-BIH (métadonnées et annotations WFDB).

### Scripts

- `scripts/download_mitbih.py` : téléchargement MIT-BIH depuis PhysioNet via `wfdb`.

### Code source

- `src/__init__.py` : package marker.
- `src/train.py` : point d'entrée principal.
  - mode `gnn` : entraînement GraphSAGE.
  - sauvegarde du meilleur modèle dans `weights/gnn_best.pth` (meilleur `f1_macro` de validation).

#### Chargement et prétraitement

- `src/data/loader.py` : utilitaires génériques ECG (QRS, extraction battements).
- `src/data/mitbih_loader.py` : lecture MIT-BIH (`wfdb`), annotations, extraction battements annotés.
- `src/data/preprocessing.py` : features de battement, segmentation, construction de graphes (`x`, `edge_index`), résumé de segment.

#### Modèles

- `src/models/gnn_models.py` : implémentation de l'architecture GNN (`GraphSAGEModel`).

### Expérimentation

- `notebooks/3_evaluate_gnn.ipynb` : évaluation du checkpoint entraîné, rapport de classification, matrice de confusion, t-SNE, visualisation de graphe.

### Poids

- `weights/gnn_best.pth` : meilleur checkpoint (sauvegardé pendant l'entraînement GNN).

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Pour `torch-geometric`, installez la version compatible avec votre version de `torch` et votre plateforme (CPU/GPU).

## Entraînement

### Entraînement GNN sur MIT-BIH

```powershell
& C:/Users/HP/miniconda3/python.exe e:/projet/coeur/src/train.py --mode gnn --data-dir data/mitbih --max-records 0 --epochs 100 --batch-size 16
```

Sorties attendues:

- logs par époque (`loss`, `val_acc`, `val_f1_macro`),
- sauvegarde du meilleur modèle dans `weights/gnn_best.pth`.

## Évaluation dans le notebook

Ouvrir `notebooks/3_evaluate_gnn.ipynb` puis exécuter les cellules dans l'ordre.

- La cellule de configuration construit des chemins absolus vers:
  - `data/mitbih`
  - `weights/gnn_best.pth`
- Si le checkpoint existe, il est chargé automatiquement.
- Le notebook affiche:
  - rapport de classification,
  - matrice de confusion,
  - projection t-SNE,
  - visualisation d'un graphe de battements.

## Limites actuelles

- Forte sensibilité au déséquilibre de classes normal/arythmie.
- Validation faite via split de segments (pas encore validation patient-wise stricte).
- Peu d'analyse d'incertitude/calibration pour le moment.

## Référence du rapport scientifique

Voir `PAPIER_MODELE_INITIAL.md` pour une version structurée type article (méthodes, résultats, limites, discussion critique).
