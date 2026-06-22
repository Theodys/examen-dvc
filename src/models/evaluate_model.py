"""
Script d'évaluation du modèle entraîné.
Calcule les métriques MSE, RMSE, MAE et R2, et génère les prédictions.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_PATH = os.path.join("data", "processed_data")
MODELS_PATH = "models"
METRICS_PATH = "metrics"


def main():
    X_test_scaled = pd.read_csv(os.path.join(PROCESSED_PATH, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_PATH, "y_test.csv")).values.ravel()

    model = joblib.load(os.path.join(MODELS_PATH, "trained_model.pkl"))

    y_pred = model.predict(X_test_scaled)

    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    scores = {"mse": round(mse, 4), "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
    print("Résultats de l'évaluation :")
    for metric, value in scores.items():
        print(f"  {metric.upper()} : {value}")

    os.makedirs(METRICS_PATH, exist_ok=True)
    with open(os.path.join(METRICS_PATH, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    predictions_df = pd.DataFrame({"y_true": y_test, "y_pred": y_pred})
    predictions_df.to_csv(os.path.join("data", "predictions.csv"), index=False)

    print(f"Métriques sauvegardées dans {METRICS_PATH}/scores.json")
    print("Prédictions sauvegardées dans data/predictions.csv")


if __name__ == "__main__":
    main()
