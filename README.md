# Examen DVC - Modélisation de la concentration de silice

## Description

Projet MLOps utilisant DVC et DagsHub pour le versioning des données et des modèles.
Le dataset contient des paramètres opérationnels du processus de flottation
des minéraux, avec pour objectif la prédiction de la concentration de silice.

## Structure du projet

```
examen-dvc/
├── data/
│   ├── raw_data/                   # Données brutes (suivi DVC)
│   │   └── raw.csv
│   └── processed_data/             # Données traitées (suivi DVC)
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       ├── X_train_scaled.csv
│       └── X_test_scaled.csv
├── metrics/
│   └── scores.json                 # Métriques d'évaluation (MSE, RMSE, MAE, R2)
├── models/
│   ├── best_params.pkl             # Meilleurs hyperparamètres (GridSearch)
│   ├── scaler.pkl                  # Scaler de normalisation
│   └── trained_model.pkl           # Modèle entraîné
├── src/
│   ├── data/
│   │   ├── split_data.py           # Téléchargement + split train/test
│   │   └── normalize_data.py       # Normalisation StandardScaler
│   └── models/
│       ├── grid_search.py          # GridSearch des hyperparamètres
│       ├── train_model.py          # Entraînement du modèle
│       └── evaluate_model.py       # Évaluation et prédictions
├── dvc.yaml                        # Définition de la pipeline DVC
├── dvc.lock                        # Verrou de la pipeline (généré)
├── requirements.txt                # Dépendances Python
└── README.md
```

## Installation et exécution

```bash
# 1. Créer et activer l'environnement virtuel
python -m venv env
source env/bin/activate        # Linux/macOS
# env\Scripts\activate         # Windows

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Initialiser DVC (si pas encore fait)
dvc init

# 4. Exécuter toute la pipeline
dvc repro

# 5. Voir les métriques
dvc metrics show

# 6. Voir le DAG de la pipeline
dvc dag
```

## Pipeline DVC

La pipeline comporte 5 étapes :

1. **split** : Téléchargement des données et split 80/20
2. **normalize** : Normalisation StandardScaler des features
3. **grid_search** : Recherche des meilleurs hyperparamètres (RandomForestRegressor)
4. **train** : Entraînement du modèle avec les paramètres optimaux
5. **evaluation** : Évaluation (MSE, RMSE, MAE, R2) et génération des prédictions

## Données

Source : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv

Le dataset contient 1817 entrées avec les variables suivantes :
- `ave_flot_air_flow` : Débit d'air moyen dans le processus de flottation
- `ave_flot_level` : Niveau moyen dans les cellules de flottation
- `iron_feed` : Quantité de minerai de fer entrant
- `starch_flow` : Débit d'amidon utilisé comme réactif
- `amina_flow` : Débit d'amine utilisé comme collecteur
- `ore_pulp_flow` : Débit de la pulpe de minerai
- `ore_pulp_pH` : Niveau de pH de la pulpe de minerai
- `ore_pulp_density` : Densité de la pulpe de minerai
- `silica_concentrate` : Concentration de silice (variable cible)
