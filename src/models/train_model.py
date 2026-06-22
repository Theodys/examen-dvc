"""
Script d'entraînement du modèle avec les meilleurs paramètres du GridSearch.
Sauvegarde le modèle entraîné au format .pkl dans le dossier models.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

PROCESSED_PATH = os.path.join("data", "processed_data")
MODELS_PATH = "models"


def main():
    X_train_scaled = pd.read_csv(os.path.join(PROCESSED_PATH, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_PATH, "y_train.csv")).values.ravel()

    best_params = joblib.load(os.path.join(MODELS_PATH, "best_params.pkl"))
    print(f"Entraînement avec les paramètres : {best_params}")

    model = RandomForestRegressor(random_state=42, **best_params)
    model.fit(X_train_scaled, y_train)

    model_path = os.path.join(MODELS_PATH, "trained_model.pkl")
    joblib.dump(model, model_path)
    print(f"Modèle entraîné et sauvegardé dans {model_path}")


if __name__ == "__main__":
    main()
