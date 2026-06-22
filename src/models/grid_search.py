"""
Script de GridSearch pour trouver les meilleurs hyperparamètres.
Modèle utilisé : RandomForestRegressor.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

PROCESSED_PATH = os.path.join("data", "processed_data")
MODELS_PATH = "models"

PARAM_GRID = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
}


def main():
    X_train_scaled = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_PATH, "y_train.csv")).values.ravel()

    print("Lancement du GridSearch (RandomForestRegressor)...")
    print(f"Paramètres testés : {PARAM_GRID}")

    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=PARAM_GRID,
        cv=3,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_scaled, y_train)

    best_params = grid_search.best_params_
    print(f"Meilleurs paramètres : {best_params}")
    print(f"Meilleur score (neg MSE) : {grid_search.best_score_:.4f}")

    os.makedirs(MODELS_PATH, exist_ok=True)
    joblib.dump(best_params, os.path.join(MODELS_PATH, "best_params.pkl"))
    print(f"Paramètres sauvegardés dans {MODELS_PATH}/best_params.pkl")


if __name__ == "__main__":
    main()
